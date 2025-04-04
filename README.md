# Практическая работа

---

## Описание: 

В условиях большого количества данных требуется написать масштабируемый алгоритм для анализа датасета и определения доминирующих и доминируемых строк по принципу  нахождения строк, где все признаки больше/меньше всех признаков другой строки. Помимо этого, провести эксперимент на предмет наличия/отсутствия свойства доминантности при сжатии (кодирования) и разжатия (декодирования) датасета.

## Цель работы:

- Считать датасет, убрать пропущенные значения и отфильтровать признаки, не соответствующие нужному типу (требуется для корректного сравнения)
- Написать функцию для нахождения доминирующих/доминируемых строк
- Написать функцию для сжатия и разжатия датасета
- Выборочно проверить строки на предмет корректности работы алгоритма
- Сравнить результаты до сжатия/разжатия и после

## Ход работы:

Подробные шаги описаны в папке `source` в файлах `find_dominated_rows_test.ipynb` ( где описан тестовый вариант для определения корректности алгоритма и сравнения строк ) и `find_dominated_rows.ipynb` (где описан полный вариант вместе со сжатием и расжатием датасета ) 

Для тестов использовался сгенерированный датасет с меньшим количестовом признаков и гарантированным количеством строк, где все элементы будут меньше всех элементов других строк ( для экономии времени тестирования ).

Выводы алгоритмов можно посмотреть соответственно в папке `output` ( `VAR_1.txt` и `VAR_2.txt` )

## Выводы

В ходе исполнения написанного алгоритма, было выявлено опытным путём присутствие признаков доминантности до сжатия и разжатия и после. Было описано два варианта алгоритмов и вывод результатов для определения корректности работы алгоритмов.

Сами датасеты можно посмотреть в папке `data`