# Инструкция по поднятию микросервиса:

---

###### Создание директории, в которую будем класть датасет:

```sh
mkdir data
```

###### Создание виртуального окружения

```py
python -m venv venv
```

###### Активация окружения

```py
source venv/bin/activate
```

###### Установка зависимостей

```py
pip install torch numpy pandas scikit-learn
```

> Сервис готов к работе

###### Тренировка модели

```py
python train.py
```
###### Оценка и вывод результатов

```py
python evaluate.py
```
