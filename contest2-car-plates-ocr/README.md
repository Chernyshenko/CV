Контест:
https://www.kaggle.com/c/car-plates-ocr-made

Основные модели взяты из бейзлайна Алексея Ярошенко - MaskRCNN (maskrcnn_resnet50_fpn)- для детекции bounding box и mask, и CRNN для распознавания.

Перевел тексты к одному словарю: латиница, верхний регистр.

В качестве аугментаций пробовал Rotate и Pad из основного бейзлайна. 
Варьировал размеры батча и число эпох

Submit:

![screen](https://github.com/Chernyshenko/CV/blob/master/contest2-car-plates-ocr/submission_sreen.png)
