### Анализ полученной программы
На мой взгляд, хотя программа и получилась рабочая т.е. она действительно находит и помечает людей на видео, однако работает она неидеально. Во-первых, программа основана на object detection, то есть покадровом поиске объектов. Использование инструментов object tracking может дать более впечатляющие результаты. Во-вторых, некоторые параметры модели стоит подбирать под конкретные задачи, в данном случае параметров чтения кадров хватает для поиска людей на видео из примера. Но в то же время, если рассматривать видео, где люди видны лишь на отдаленности, то результаты программы могут быть хуже. 
### Возможные улучшения
1) Реализовать аналогичную программу, но с использование инструментов object tracking, например, TrackerBoosting_create. Это улучшит "соединяемость" одного объекта на разных кадрах.
2) Реализовать отдельную программу для каждой отдельной задачи поиска людей с разными параметрами для чтения и обработки кадров.
