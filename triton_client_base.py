from abc import ABC, abstractmethod
from typing import List
import numpy as np


class TritonClient(ABC):
    """
    Абстрактный класс для создания клиента для работы
    на Triton Inference Server
    """

    @abstractmethod
    def connect(self):
        """
        Абстрактный метод для подключения
        клиента к Triton Inference Server
        """
        pass


    @abstractmethod
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Абстрактный метод для предобработки массива
        """
        pass

    @abstractmethod
    def infer(self, input_imgs: List[str]):
        """
        Абстрактный метод для инференса
        """
        pass

    @abstractmethod
    def _postprocess(self, output_data: np.ndarray):
        """
        Абстрактный метод для постпроцессинга
        """
        pass


    @abstractmethod
    def disconnect(self):
        """
        Абстрактный метод для отключения от Triton Inference Server и
        очистки ресурсов
        """
        pass



