# First Task 
## Oсновы, генерация сигнала в реальном времени

# Second Task 
## Формирование сигналов, визуализация в Python. 
<img src= "Second Task/test_sort_speed.png">

# Third Task
## Изучение основных параметров библиотеки PyAdi для Adalm Pluto SDR
Определение наисильнейшего сигнала - 2412 Мгц
Передача данных на данной частоте
<img src= "Third Task/signal_data.jpg">

Временные диаграммы 
<img src= "Third Task/sampling_rate_1000.png">

Количество отсчетов сигнала порядка 200 в команде arrange
<img src= "Third Task/arrange_200.png">

# Fourth Task
## Изучение основных свойств ДПФ с помощью моделирования в  Python/Spyder
<img src= "Fourth Task/image/ds_fourier.png">
Для заданных значений частоты сигнала и частоты дискретизации получите дискретное колебание, отсчеты посмотрите в Variable Explorer. Далее увеличьте частоту сигнала в несколько раз, при этом так же увеличится и частота дискретизации, но отношение частоты сигнала и частоты дискретизации - нормированная частота останется той же величиной.
    Сравните дискретные отсчеты первого и второго сигналов.
<img src= "Fourth Task/image/1.png">
Измените частоту сигнала в целое чисто раз, определите номер точки
    ДПФ для данного сигнала.
<img src= "Fourth Task/image/3.png">
Измените количество точек ДПФ до 512. Вычислите шаг частот между
    точками ДПФ ∆f = fs/N. Определите, в какой точке ДПФ находится заданный сигнал.
<img src= "Fourth Task/image/4.png">

# Fifth Task
## Передача/прием sin() сигнала. Реализация АМ модуляции. PlutoSDR.
Передача одиночного сигнала
<img src= "Fifth Task/image/image1.jpg">

Передача двоичного кода символа   
<img src= "Fifth Task/image/image2.jpg">

# Sixth Task 
## Модуляции QPSK, QAM. Раздельный приём и передача на SDR
Моделирование сигнала, накладывание шума, декодирование
<img src= "Sixth Task/img/base/1.png">
Отправленный и полученный сигнал соответственно
<img src= "Sixth Task/img/1.jpg">
спектр полученного сигнала
<img src= "Sixth Task/img/2.jpg">

# Курсовая работа
## Определение параметров сигнала OFDM и моделирование приема OFDM в радиоканале
Созвездия на поднесущих на выходе канала (до коррекции)
<img src= "Coursework/img/1.png">
Частотная характеристика канала (АЧХ)
<img src= "Coursework/img/2.png">
QAM после коррекции
<img src= "Coursework/img/3.png">
Посчитаем Bit error rate и выведем его
<img src= "Coursework/img/4.png">


# Расчетно-графическая работа
## изучение алгоритма оценивания канала по методу наименьших квадратов и изучение влияния оценки канала на характеристики прекодирования ZF с помощью моделирования в среде Python

Диаграммы направленности АР с применением весового вектора для выбранных абонентов
<img src= "RGR/img/1.png">
Загруженный канал для 1 пользователя
<img src= "RGR/img/2.png">
Загруженный канал для 4 пользователя
<img src= "RGR/img/3.png">
Диаграмма направленности
<img src= "RGR/img/4.png">