import argparse
import asyncio as aio
import sys
from typing import Any, List, Optional
from abc import ABC

import cv2  # type: ignore
import numpy as np  # type: ignore
import tritonclient.grpc.aio as grpcclient  # type: ignore
import tritonclient.utils.shared_memory as shm  # type: ignore
from tritonclient import utils  # type: ignore

from yolo_triton_client_base import BaseYoloTritonClient
from yolo import YOLOPoseOut

class PoseYoloTritonClient(BaseYoloTritonClient):
    def __init__(
        self,
        keypoints_count: int = 17,
        **kwargs
    ):
        """
        Асинхронный gRPC Triton Inference Server клиент

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
            confidence_threshold: Порог уверенности для фильтрации
            verbose: Включить подробный вывод
        """

        prediction_size = keypoints_count * 3 + 6

        super().__init__(prediction_size=prediction_size, **kwargs)

        self.keypoints_count = keypoints_count


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

        if len(input_imgs) == 1:
            batch_tensor = batch_tensor[0]

        actual_input_byte_size = batch_tensor.nbytes

        if actual_input_byte_size > self.input_byte_size:
            raise RuntimeError(
                f"Batch data size {actual_input_byte_size} bytes exeeds shared memory size {self.input_byte_size} bytes. "
                f"Batch shape: {batch_tensor.shape}, max expected: ({self.max_batch_size}, 3, {self.input_height}, {self.input_width})"
            )

        n_batch = batch_tensor.shape[0] if batch_tensor.ndim == 4 else 1

        actual_output_byte_size = int(
            n_batch
            * self.max_detections
            * self.prediction_size
            * np.dtype(np.float32).itemsize
        )


        shm.set_shared_memory_region(self.shm_ip_handle, [batch_tensor])

        inputs = []
        inputs.append(
            grpcclient.InferInput(self.input_name, batch_tensor.shape, "FP32")
        )
        
        inputs[-1].set_shared_memory(f"{self.model_name}_input", actual_input_byte_size)

        outputs = []
        outputs.append(grpcclient.InferRequestedOutput(self.output_name))
        outputs[-1].set_shared_memory(
            f"{self.model_name}_output", actual_output_byte_size
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
            YOLOPoseOut(data=data, keypoints_data=kp, orig_shape=shape)
            for det, shape in zip(output_data, original_shapes)
            for data, kp in [self._postprocess(output_data=det, orig_shape=shape)]
        ]


    def _postprocess(self, output_data: np.ndarray, orig_shape):
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

        def _scale_kp_to_orig(kp_xy, orig_h, orig_w, inp_h, inp_w):
            '''Обратное преобразование после препроцессинга: масштабирование xyxy к оригинальному размеру'''
            gain_w = orig_w / inp_w
            gain_h = orig_h / inp_h

            out = kp_xy.copy()
            out[:, :, [0]] *= gain_w
            out[:, :, [1]] *= gain_h

            return out

        if output_data.ndim == 3:
            output_data = output_data[0]

        mask = output_data[:, 4] > self.confidence_threshold
        filtered = output_data[mask]
        n = filtered.shape[0]

        if n == 0:
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0, self.keypoints_count, 3), dtype=np.float32),
            )

        boxes = self._scale_boxes_to_orig(boxes_xyxy=filtered[ :, :6], 
                                                         orig_h=orig_shape[0],
                                                         orig_w=orig_shape[1],
                                                         inp_h=self.input_height,
                                                         inp_w=self.input_width)

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_shape[1] - 1)

        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_shape[0] - 1)

        kp_flat = filtered[:, 6 : 6 + self.keypoints_count * 3]

        keypoints = kp_flat.reshape(n, self.keypoints_count, 3).astype(np.float32)

        scaled_to_orig_kp = _scale_kp_to_orig(keypoints, 
                                              orig_h=orig_shape[0],
                                              orig_w=orig_shape[1],
                                              inp_h=self.input_height,
                                              inp_w=self.input_width)
        
        scaled_to_orig_kp[:, :, [0]] = np.clip(scaled_to_orig_kp[:, :, [0]], 0, orig_shape[1] - 1)
        scaled_to_orig_kp[:, :, [1]] = np.clip(scaled_to_orig_kp[:, :, [1]], 0, orig_shape[0] - 1)

        ids = np.full(n, -1, dtype=np.float32)

        data = np.insert(boxes, 4, ids, axis=1)
        
        return data.astype(np.float32), scaled_to_orig_kp
        
        

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

    print("Создание клиента")

    client = PoseYoloTritonClient(
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
        print("Запуск!")
        detections = await client.infer(FLAGS.path_to_image)
        print(f"Число детекций: {len(detections)}")
        
        for i, detection in enumerate(detections):
            print(f"\n{'='*20} Image {i+1} {'='*20}")
            print(f"Classes: {detection.classes}")
            print(f"IDs: {detection.ids}")
            print(f"BBoxes XYXY: {detection.bboxes.xyxy}")
            print(f"BBoxes XYXY norm: {detection.bboxes.xyxyn}")
            print(f"BBoxes XYWH: {detection.bboxes.xywh}")
            print(f"Old format: {detection.old_format}")
            print(f"Confidences: {detection.confs}")
            print(f"Keypoints: {detection.keypoints.xy}")

if __name__ == "__main__":
    aio.run(main())
