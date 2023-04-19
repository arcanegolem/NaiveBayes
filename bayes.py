import pandas as pd
import numpy as np

class NaiveBayes:
    initial_classes : np.ndarray
    initial_data    : np.ndarray

    def __init__(self, initial: pd.DataFrame) -> None:
        '''
        Инициализация наивного Баесовского классификатора
        '''
        initial = initial.values
        
        ''' Классы из таблицы '''
        self.initial_classes = initial[:, -1:]

        ''' Данные из таблицы '''
        self.initial_data    = initial[:,0:-1:]

        self.class_probability_dict = dict.fromkeys(np.unique(self.initial_classes), 1)
        self.class_frequency = np.array([])

        ''' Подсчет частоты каждого класса '''
        for key in self.class_probability_dict.keys():
            self.class_frequency = np.append(self.class_frequency, np.count_nonzero(self.initial_classes == key))
        

    def classify(self, to_classify: list) -> dict:
        '''
        Метод классификации

        to_classify - список критериев, по которым будет производится классификация
        '''

        to_classify = np.array(to_classify)

        if len(to_classify) != len(self.initial_data[0]):
            raise IndexError(f"Количество критериев отличается! ---> {len(to_classify)} != {len(self.initial_data[0])}")

        ''' Проход по каждой категории '''
        for val_idx in range(len(to_classify)):
            ''' Проверка на наличие критерия в исходниках '''
            if to_classify[val_idx] not in self.initial_data[:, val_idx]:
                raise NameError(f"Некорректное значение критерия! ---> \"{to_classify[val_idx]}\"")

            ''' Проход по классам'''
            for class_v, class_idx in zip(self.class_probability_dict, range(len(self.class_probability_dict.keys()))):
                
                ''' Нахождение полей с совпадением с исходыми данными '''
                data_match  = np.where(self.initial_data[:, val_idx] == to_classify[val_idx])[0]
                class_match = np.where(self.initial_classes == class_v)[0]

                ''' Частота встречи конкретного значения с конкретными классом '''
                condition_frequency = np.count_nonzero(np.isin(data_match, class_match) == True)

                self.class_probability_dict[class_v] *= condition_frequency/self.class_frequency[class_idx]

        for class_v, class_idx in zip(self.class_probability_dict, range(len(self.class_probability_dict.keys()))):
            self.class_probability_dict[class_v] *= self.class_frequency[class_idx] / len(self.initial_classes)
        
        ''' Нормализация '''
        for class_v in self.class_probability_dict:
            self.class_probability_dict[class_v] = self.class_probability_dict[class_v] / sum(self.class_probability_dict.values())

        return self.class_probability_dict