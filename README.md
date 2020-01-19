## Introduction
百度车道线识别

## Useage

1. 使用util下面的create_csv.py 按照7:2:1的比例生成训练集、测试集、验证集
2. 使用class_balance.py分析类别分布，防止训练集覆盖的某个类别点不够多
3. 使用train.py 进行训练
4. 使用test.py 进行验证测试集

## 配置

可以在config里修改使用那些数据

## 待做

inference 代码