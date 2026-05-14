import argparse
import asyncio as aio
import sys
from typing import Any, List, Optional

import cv2  # type: ignore
import numpy as np  # type: ignore
import tritonclient.grpc.aio as grpcclient  # type: ignore
import tritonclient.utils.shared_memory as shm  # type: ignore
from tritonclient import utils  # type: ignore

from yolo import YOLOBaseOut


class YoloTritonClient:
    def __init__(
        self,
        url: str = "localhost:8001",
        model_name: str = "yolo",
        model_version: int = 1,
        input_name: str = "images",
        output_name: str = "output0",
        max_batch_size: int = 1,
        input_width: int = 640,
        input_height: int = 640,
        max_detections: int = 300,
        confidence_threshold: float = 0.8,
        verbose: bool = False,
    ):
        """
        Класс для создания клиента для работы с моделью YOLO на сервере Triton

        Args:
            url: Адрес Triton сервера (gRPC порт 8001)
            model_name: Название модели в Triton
            model_version: Версия модели в Triton
            input_name: Название входного слоя модели
            output_name: Название выходного слоя модели
            max_batch_size: Максимальный размер батча
            input_width: Ширина входного изображения
            input_height: Высота входного изображения
            max_detections: Максимальное число детекций
            confidence_treshold: Порог уверенности для фильтрации
            verbose: Включить подробный вывод
        """

        self.url = url
        self.model_name = model_name
        self.model_version = model_version
        self.input_name = input_name
        self.output_name = output_name
        self.max_batch_size = max_batch_size
        self.input_width = input_width
        self.input_height = input_height
        self.max_detections = max_detections
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose

        self.single_frame_size = (
            3 * input_height * input_width * np.dtype(np.float32).itemsize
        )
        self.input_byte_size = self.single_frame_size * max_batch_size
        self.output_byte_size = (
            max_batch_size * max_detections * 6 * np.dtype(np.float32).itemsize
        )

        self.client: Optional[grpcclient.InferenceServerClient] = None
        self.shm_ip_handle: Any = None
        self.shm_op_handle: Any = None
        self._is_connected = False

    async def __aenter__(self):
        """
        Асинхронный вход в контекст
        """
        await self.connect()
        return self


    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Асинхронный выход из контекста
        """

        await self.disconnect()
        return False

    async def connect(self):
        """
        Подключение к Triton Inference Server
        """
        if self._is_connected:
            if self.verbose:
                print("Already connected, skipping")
            return

        try:
            self.client = grpcclient.InferenceServerClient(
                url=self.url, verbose=self.verbose
            )
        except Exception as e:
            print("Failed to connect to Triton Server: " + str(e))
            sys.exit()

        await self.client.unregister_system_shared_memory()
        await self.client.unregister_cuda_shared_memory()

        self.shm_op_handle = shm.create_shared_memory_region(
            f"{self.model_name}_output",
            f"/{self.model_name}_output",
            self.output_byte_size,
        )

        await self.client.register_system_shared_memory(
            f"{self.model_name}_output",
            f"/{self.model_name}_output",
            self.output_byte_size,
        )

        self.shm_ip_handle = shm.create_shared_memory_region(
            f"{self.model_name}_input",
            f"/{self.model_name}_input",
            self.input_byte_size,
        )

        await self.client.register_system_shared_memory(
            f"{self.model_name}_input",
            f"/{self.model_name}_input",
            self.input_byte_size,
        )

        self._is_connected = True
        if self.verbose:
            print(f"Connected to {self.url}, model: {self.model_name}")


    async def disconnect(self) -> None:
        """
        Отключение и очистка ресурсов.

        """
        if not self._is_connected:
            return

        if self.client:
            await self.client.unregister_system_shared_memory()
            await self.client.unregister_cuda_shared_memory()

            if self.shm_ip_handle:
                shm.destroy_shared_memory_region(self.shm_ip_handle)
            if self.shm_op_handle:
                shm.destroy_shared_memory_region(self.shm_op_handle)

            await self.client.close()
            self.client = None

        self._is_connected = False
        if self.verbose:
            print("Disconnected")


    async def _preprocess(self, img: np.ndarray):
        """
        Метод для предобработки массива:
        - resize(self.input_width, self.input_height)
        - BGR -> RGB
        - HWC -> CHW
        - нормализация

        Args:
            img: np.ndarray массив изображения

        Returns:
            img.astype(np.float): массив обработанного изображения
        """

        img = cv2.resize(img, (self.input_width, self.input_height))

        img = img[:, :, ::-1]

        img = np.transpose(img, (2, 0, 1)) / 255.0

        return img.astype(np.float32)

   
    async def infer(self, input_imgs: List[str]):
        """
        Метод для инференса с выходом YOLOBaseOut

        Args:
            input: List[str] список путей к изображениям

        Returns:
            out: Массив детекций List[YOLOBaseOut]

        """
        if not self._is_connected:
            raise RuntimeError("Client not connected. Call connect() first")
        
        if not input_imgs:
            raise ValueError("Empty image list provided")


        if len(input_imgs) > self.max_batch_size:
            raise ValueError(f"Batch size {len(input_imgs)} exceeds {self.max_batch_size}")
        
        original_shapes = []
        images_tensors = []
        for path in input_imgs:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Couldn't load image: {path}")
            original_shape = img.shape[:2]
            tensor = await self._preprocess(img)
            images_tensors.append(tensor)
            original_shapes.append(original_shape)
        batch_tensor = np.stack(images_tensors, axis=0)

        actual_input_byte_size = batch_tensor.nbytes

        if actual_input_byte_size > self.input_byte_size:
            raise RuntimeError(
                f"Batch data size {actual_input_byte_size} bytes exeeds shared memory size {self.input_byte_size} bytes. "
                f"Batch shape: {batch_tensor.shape}, max expected: ({self.max_batch_size}, 3, {self.input_height}, {self.input_width})"
            )

        shm.set_shared_memory_region(self.shm_ip_handle, [batch_tensor])

        inputs = []
        inputs.append(
            grpcclient.InferInput(self.input_name, batch_tensor.shape, "FP32")
        )
        
        inputs[-1].set_shared_memory(f"{self.model_name}_input", self.input_byte_size)

        outputs = []
        outputs.append(grpcclient.InferRequestedOutput(self.output_name))
        outputs[-1].set_shared_memory(
            f"{self.model_name}_output", self.output_byte_size
        )

        results = await self.client.infer(
            model_name=self.model_name, inputs=inputs, outputs=outputs
        )

        output_meta = results.get_output(self.output_name)

        if output_meta is None:
            raise RuntimeError(f"Output '{self.output_name}' not found in response")

        output_data = shm.get_contents_as_numpy(
            self.shm_op_handle,
            utils.triton_to_np_dtype(output_meta.datatype),
            output_meta.shape,
        )

        return [
            YOLOBaseOut(data= await self._postprocess(det), orig_shape=original_shape)
            for det, original_shape in zip(output_data, original_shapes)
        ]
    

    async def _postprocess(self, output_data: np.ndarray):
        """
        Постобработка выхода модели: 
        - вставка id (нет трекера)
        - фильтрация по confidence
        - клип по размерам изображения

        Args:
            output_data: np.ndarray выход модели
        
        Returns:
            data_with_ids: np.ndarray массив,
            отфильтрованный и подготовленный к YOLOBaseOut
        """
        confs_mask = output_data[..., 4] > self.confidence_threshold

        data_filtered_by_confs = output_data[confs_mask]

        data_filtered_by_confs[:, [0, 2]] = np.clip(data_filtered_by_confs[:, [0, 2]], 0, self.input_width)
        data_filtered_by_confs[:, [1, 3]] = np.clip(data_filtered_by_confs[:, [1, 3]], 0, self.input_height)
        data_with_ids = np.insert(data_filtered_by_confs, 4, np.array([-1] *  data_filtered_by_confs.shape[0]), axis=1)


        return data_with_ids


FLAGS = None


async def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )
    parser.add_argument(
        "-m", "--model-name", type=str, required=True, help="Model name"
    )
    parser.add_argument(
        "-e",
        "--model-version",
        type=str,
        required=False,
        default="",
        help="Model version",
    )
    parser.add_argument(
        "-i",
        "--path-to-image",
        nargs="+",
        type=str,
        required=True,
        help="Path to image",
    )
    parser.add_argument(
        "-conf",
        "--confidence-threshold",
        type=float,
        required=False,
        default=0.5,
        help="Confidence threshold",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        required=False,
        help="Batch size",
    )

    FLAGS = parser.parse_args()

    client = YoloTritonClient(
        url=FLAGS.url,
        model_name=FLAGS.model_name,
        max_batch_size=FLAGS.batch_size,
        input_width=640,
        input_height=640,
        max_detections=300,
        confidence_threshold=FLAGS.confidence_threshold,
        verbose=FLAGS.verbose,
    )

    async with client:        
        detections = await client.infer(FLAGS.path_to_image)
        
        for i, detection in enumerate(detections):
            print(f"\n{'='*20} Image {i+1} {'='*20}")
            print(f"Classes: {detection.classes}")
            print(f"IDs: {detection.ids}")
            print(f"BBoxes XYXY: {detection.bboxes.xyxy}")
            print(f"BBoxes XYXY norm: {detection.bboxes.xyxyn}")
            print(f"BBoxes XYWH: {detection.bboxes.xywh}")
            print(f"Old format: {detection.old_format}")
            print(f"Confidences: {detection.confs}")


if __name__ == "__main__":
    aio.run(main())
