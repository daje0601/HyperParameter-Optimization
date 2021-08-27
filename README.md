## **🔶 목차**

1. **예측모델 관리 및 성능 고도화**
- 시나리오
- 데이터 안내
- 분석 내용
- 튜닝 내용
- 결론



## 🔶시나리오

> 데이터 사이언티스트로 입사를 하게 되었습니다. 
팀장님께서 기존 수요 예측 모델의 성능을 개선하라는 첫 업무를 주셨습니다. 

이에, 분석 목적에 맞는 머신러닝 모델을 선정하고, 
기준모델 설정 및 Hyperparameter Tuning을 통하여 성능을 개선하고자 합니다.

## 🔶 데이터 안내



해당 데이터는 현재 kaggle에서 competition이 진행되고 있는 데이터입니다.
실제 대출 채무 불이행과 관련된 손실 계산 데이터로써, feature name은 f1, f2 등의 형식으로 암호화 되어 있으며, feature의 수는 f0~f99까지 100개의 featur로 구성되어 있습니다. 해당 competition은 구글에서 하이퍼파라미터 튜닝 실력을 검증하기 위한 목적으로 진행하고 있습니다. 

**평가 지표 : RMSE(root mean squared error)**

**분석 목적 : 하이퍼파라미터 튜닝을 통한 성능 극대화** 

**선정 사유 : 입사 후 예측모델 관리 및 성능 고도화 업무가 중요하다 생각하였습니다. 
                  이에, 하이퍼파라미터 튜닝을 보여드리고자 별도의 페이지로 구성하였습니다.** 

( 그 외 데이터를 꼼꼼하게 살피는 프로젝트는 상단에 포트폴리오 링크를 참조 부탁 드립니다. )

## 🔶 분석 내용



1. **전처리** 
- 데이터 타입, 결측치, 고유값 수를 확인한 결과, 해당 데이터는 카테코리형 데이터가 모두 숫자형 데이터로 치환이 되어 있습니다. 이에, 그래프를 통해 어떤 feature 카테코리형인지 확인해보도록 하겠습니다.
- 여기서 활용할 그래프는 Kdeplot(kdeplot : kernel density estimation )입니다.
- density estimation(밀도추정) : (관측된) 데이터들의 분포로부터 원래 변수의 (확률) 분포 특성을 추정하고자 하는 것을 의미합니다.
- 데이터에 대한 사전 정보를 아는 것은 거의 없기 때문에 사전 정보나 지식 없이 순수하게 관측된 데이터만으로 확률밀도함수를 추정해야 하는데 이를 non-parametric density estimation라 부릅니다. non-parametric density estimation의 대표적인 예가 히스토그램인데 히스토그램은 아래와 같이 3가지의 단점이 있습니다.

       1) bin의 경계에서 불연속성이 나타난다는 점,

       2) bin의 크기 및 시작 위치에 따라서 히스토그램이 달라진다는 점,

       3) 고차원(high dimension) 데이터에는 메모리 문제 등으로 사용하기 
          힘들다는 점 등의 문제점을 갖습니다.

- 이러한 문제점을 해결하기 위한 그래프가 kdeplot입니다.
- 여기서 각각의 feature마다 그래프가 그려지며, 2봉, 3봉으로 출력이 되는 그래프는 카테고리형으로 추정할 수 있습니다.

1. **target 데이터 분석** 

![image](https://user-images.githubusercontent.com/73736988/129168151-2b12f7d1-c292-421c-a459-a9af8f19159f.png)

- 데이터의 Positive skew(왼쪽으로 치우진 모양)을 보이고 있습니다.
- target 데이터도 kdeplot과 코드로 살펴보겠습니다.

![image](https://user-images.githubusercontent.com/73736988/129168167-95c77452-79de-4876-84f4-11cd83c2ecf5.png)


```jsx
# 5달러의 loss를 발생시킨 인원을 확인하기 위한 코드 
train.loss[train["loss"] <= 5].count() / train["loss"].count(), # 59%

# 5달러의 loss를 발생시킨 금액 합계 비율을 확인하기 위한 코드 
train.loss[train["loss"] <= 5].sum() / train["loss"].sum() # 13%
```

- 전체 인원의 약 60%는 5달러 미만으로 Loss를 발생 시켰으며, 전체 Loss에 13%에 해당 합니다.
이는, 큰 금액으로 대출을 진행한 소수의 고객이 회사에 손실을 입히는 경향을 관측할 수 있었습니다.
- 예상했던 것과 비슷하게 봉우리가 여러 개인 feature가 다수 확인되었습니다. 이들은 카테고리형 데이터일 확률이 있습니다. 그러나, 해당 데이터는 고객의 요청에 따라 숫자로 치환되었으며, 일종의 암호화되어 있습니다. 이를, 다시 라벨링 하는 작업은 오히려 데이터 분석하는데 방해된다고 생각하여 이 상태 그대로 모델링을 진행해보겠습니다.

## 🔶모델링



### 1. **Base Model**

- 모델명 : LinearRegressor
- RMSE : 62.02830718991018
- 모델 선정 사유 : 평가지표가 회귀용 지표이기에 가장 간단한 LinearRegressor모델을 선정함

### 2. **Main Model**

- 모델명 : LightGBMRegressor
- RMSE : 62.02830718991018
- 모델 선정 사유 :

    1) 실험 데이터가 10,000이상 row를 가지고 있습니다.

    2) 학습속도가 빠른 알고리즘이 필요하였습니다.

    3) Light GBM은 leaf-wise인 반면 xgboost 등 다른 알고리즘은 level-wise이여서 과적합되기 쉽습니다.

    4) 즉, 학습이 진행될수록max delta loss를 가진 leaf를 선택하게 되게 됩니다. 동일한 leaf를 확장할 때,leaf-wise 알고리즘은 level-wise 알고리즘보다 더 많은 loss, 손실을 줄일 수 있습니다.

- 진행방법

    1) 튜닝하지 않고, StratifiedKFold 적용한 성능 확인합니다. 

    -. StratifiedKFold는 k의 fold로 나누어질 때 타겟의 비율을 일정하게 유지해주는 역할을 합니다.

    2) 오토 튜닝을 진행하여 최적의 하이퍼파라미터를 찾습니다. 

        아래와 같이 과적합을 방지하기 위한 파라미터 위주로 튜닝을 진행하였습니다. 

    ```jsx
    params = {
    # 평가 방식이 rmse이므로 해당 metrics 사용 
    "metric": "RMSE",
    # 학습하는 과정을 생략하기 위해서 기재함        
    "verbosity": -1,  
    # gdbt, rf, dart, goss 중 gbdt는 정확성, 효율성 및 안정성 때문에 사용함
    "boosting_type": "gbdt", 
    # 학습속도가 빠른 대신 과적합 될 위험이 있으므로 사용
    "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
    # 학습속도가 빠른 대신 과적합 될 위험이 있으므로 사용
    "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    # 이를 통해 각 weak_model이 가질 수 있는 최대 리프 수를 설정
    "num_leaves": trial.suggest_int("num_leaves", 2, 256),
    # feature 샘플링을 진행하여 과적합을 방지함 
    "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
    # 리샘플링 없이 데이터의 일부를 무작위로 선택하는 코드
    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
    # 배깅을 활성화하기 위한 코드
    "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
    # weak_model 하나에 가질 수 있는 leaf의 수를 제한하는 코드 
    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    ```

    여기서 활용되는 optuna를 선택한 이유는 아래 그림에서 와 같이 optuna가 우수한 성능을 보였기 때문에 최적의 하이퍼파라미터를 찾는데 적절하다고 생각하였습니다. 

![image](https://user-images.githubusercontent.com/73736988/129168215-2dc40ccf-30e0-466a-afb7-2111c02284d7.png)

  
    - 이미지 reference 출처 :

        (1) Hutter, F., Hoos, H., Leyton-Brown, K .: 하이퍼 파라미터 중요성 평가를위한 효율적인 접근 방식. 에서 : Xing and Jebara [157], 754–762 (2014)

        (2) Bergstra, J., Bengio, Y .: 하이퍼 매개 변수 최적화를위한 무작위 검색. 기계 학습 연구 저널 13, 281–305 (2012)

    3) 최적의 파라미터와 StratifiedKFold를 적용하여 최적의 성능을 찾습니다.

    ```jsx
    Best trial: 20
    	with value: 7.87168
    	Best params:
    		n_estimators: 2000
    		reg_lambda: 100
    		reg_alpha: 7
    		subsample: 1.0
    		colsample_bytree: 0.30000000000000004
    		max_depth: 9
    		min_child_weight: 10
    		gamma: 18.33239951135272
    ```

### 3. 시각화

![image](https://user-images.githubusercontent.com/73736988/129168234-f6f30ced-84aa-4ed5-ae24-d861aada7c67.png)

1) optuna를 시각화하여 각각의 파라미터간의 관계를 살펴보았습니다. 역시 예상했던 것처럼 과적합을 방지하는 gamma, alpha 파라미터가 일정한 루트를 보였습니다.

2) 또한, 가장 큰 영향을 파라미터는 n_estimators와 max_depth 였으며, 
![image](https://user-images.githubusercontent.com/73736988/129168247-a56404ff-5673-498f-a8d7-f33d4be995ef.png)

3) eli5를 활용하여 가장 큰 영향을 준 feature를 확인결과, f52, f81으로 확인되었습니다. 

![image](https://user-images.githubusercontent.com/73736988/129168259-ed23ec3c-5134-475d-92a3-a879ee9df0a1.png)

![image](https://user-images.githubusercontent.com/73736988/129168271-0aa38b9a-b4cc-4854-a658-78bf68da3e03.png)  

eli5가 정확하게 측정하지 못하였을 수도 있으니, shap 라이브러리를 활용하여 재확인토록 하겠습니다. 역시, f52가 찐한 빨간색을 띄우며, 우측으로 도드라져 있습니다.(빨간색은 성능을 향상시킨 요인을 표기하는 것이며, 파란색은 성능을 하락시킨 요인을 표기하는 것 입니다.)

### 4. **성능**

- **담당자 max rmse : 7.87261**
- **kaggle max rmse : 7.85761**
- kaggle max rmse와 비교하였을 때, 준수한 성능으로 하이퍼 파라미터 튜닝을 완료하였습니다.

### 5. 결론

입사하여 위와 같은 라이브러리를 활용하여 예측모델 성능 고도화 업무를 차질 없이 임하도록 하겠습니다. 감사합니다.
