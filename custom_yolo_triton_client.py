import argparse
import asyncio as aio
import sys
from builtins import range
from ctypes import *
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
        confidence_threshold: float = 0.5,
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

    # Метод для подключения к серверу Triton

    async def connect(self):
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

    # Метод для отключения от сервера Triton

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

        img = cv2.resize(img, (self.input_width, self.input_height))

        img = img[:, :, ::-1]

        img = np.transpose(img, (2, 0, 1)) / 255.0

        return img.astype(np.float32)

    async def infer_image(self, input_image_path: str):
        """
        Инференс из уже подготовленного тензора.

        Args:
            input_image_path: путь к изображению

        Returns:
            Массив детекций формы (max_detections, 6)
        """
        if not self._is_connected:
            raise RuntimeError(
                "Client not connected. Call connect() first ot use asyc with"
            )
        
        img = cv2.imread(input_image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {input_image_path}")
        
        original_height, original_width = img.shape[:2]
        
        input_tensor = await self._preprocess(img)

        actual_input_byte_size = input_tensor.nbytes

        if actual_input_byte_size > self.input_byte_size:
            raise RuntimeError(
                f"Batch data size {actual_input_byte_size} bytes exeeds shared memory size {self.input_byte_size} bytes. "
                f"Batch shape: {input_tensor.shape}, max expected: ({self.max_batch_size}, 3, {self.input_height}, {self.input_width})"
            )

        shm.set_shared_memory_region(self.shm_ip_handle, [input_tensor])

        inputs = []
        inputs.append(grpcclient.InferInput(self.input_name, input_tensor.shape, "FP32"))
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
        )[0]

        data_with_ids = await self._postprocess(output_data)

        out = YOLOBaseOut(orig_shape=(original_height, original_width), data=data_with_ids)

        return out

    async def infer_batch(self, input_batch: List[str]):
        """
        Метод для инференса батча с выходом YOLOBaseOut

        Args:
            input_batch: List[str] список путей к изображениям

        Returns:
            Массив детекций List[YOLOBaseOut]

        """
        if len(input_batch) > self.max_batch_size:
            raise ValueError(f"Batch size {len(input_batch)} exceeds {self.max_batch_size}")
        
        original_shapes = []
        images_tensors = []
        for path in input_batch:
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
        data_with_ids = np.insert(output_data, 4, np.array([-1] *  output_data.shape[0]), axis=1)

        confs_mask = data_with_ids[..., 5] > self.confidence_threshold

        data_filtered_by_confs = data_with_ids[confs_mask]

        data_filtered_by_confs[:, [0, 2]] = np.clip(data_filtered_by_confs[:, [0, 2]], 0, self.input_width)
        data_filtered_by_confs[:, [1, 3]] = np.clip(data_filtered_by_confs[:, [1, 3]], 0, self.input_height)

        return data_filtered_by_confs


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

    try:
        await client.connect()

        image_paths = FLAGS.path_to_image
        if len(image_paths) > 1:
            detections = await client.infer_batch(image_paths)
            for i in range(len(image_paths)):
                print(detections[i].classes)
                print(detections[i].ids)
                print(detections[i].bboxes.xyxy)
                print(detections[i].bboxes.xyxyn)
                print(detections[i].bboxes.xywh)
                print(detections[i].old_format)
                print(detections[i].confs)
                print("\n============================\n")
        else:
            detections = await client.infer_image(image_paths[0])

            print(detections.classes)
            print(detections.ids)
            print(detections.bboxes.xyxy)
            print(detections.bboxes.xyxyn)
            print(detections.bboxes.xywh)
            print(detections.old_format)
            print(detections.confs)
    finally:
        await client.disconnect()


if __name__ == "__main__":
    aio.run(main())
