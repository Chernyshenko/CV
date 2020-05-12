contest:
https://www.kaggle.com/c/made-thousand-facial-landmarks

run:
python hack_train.py --name "test" --data "data" --gpu --batch-size 64 --epochs 20 --learning-rate 0.01

В решении просто используется модель resnet101
Простые аугментации (яркость\контраст, размытие) не принесли профита

[screen](https://github.com/Chernyshenko/CV/blob/master/contest1-facial-landmarks/submissions.png)
