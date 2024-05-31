# IS team project – 랜덤 포레스트 구현 관련 코드

# 의도: 라이브러리를 사용하지 않고, 랜덤 포레스트를 구성하는 Decision Tree와 랜덤 포레스트의 핵심 알고리즘을 평문 코드로 구현.
# 이후 동형암호를 이용한 데이터 연산이 가능하도록 변경하는 것을 목적으로 한다.

# 우리의 목적은 random forest 알고리즘을 통해 주어진 데이터에서 feature_importance를 추출, 해당 가중치를 바탕으로 점수 예측 식을 구성하는 것이다.
# 그래서, 흐름은 1. decision tree 구성 - 2. random forest 구성 - 3. feature importance 추출 - 4. 실제 데이터 투입을 통한 모델 test 가 된다.



class RandomForest: # decision tree를 통해 생성된 다수의 tree를 이용하는 경우의 코드에 해당함 - 해당 클래스와 연결됨
   def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None): # 초기 설정(parameter 데이터 input)
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

   def fit(self, X, y): # 주어진 데이터셋에 맞춰 tree를 비교, 그리고 지금까지 제시된 다른 tree와 비교하여 어떤 부분에 속할지를 말하게 된다.
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_samp, y_samp = self._bootstrap_samples(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

   def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
   # 아하.. 이 부분은 완벽하지 않으나, 앙상블의 장점? 효과를 보기 위해서 삽입해야 했던 method에 해당함.
   # 여튼 데이터셋에서 n_samples개의 데이터를 무작위로 추출하여 트리를 정해진 횟수(위의 경우에서는 tree의 개수)만큼 훈련하여 앙상블 모델을 구현하려 함.
   # 사실 여기는 구동 과정을 잘 몰라서 gpt 기반으로 핵심 부분을 추출해달라고 부탁했는데, 아래의 코드를 추출하였음
   # 이게 어느 정도까지 작동을 해주는 지는 잘 모르겠으나, 해당 메소드는 random forest의 구현을 위해서는 필수적인 역할을 함.

   def _most_common_label(self, y): # random forest를 사용하기 때문에, tree 별로 비교를 거쳤을 때, 결론적으로 가장 많이 등장한(효과적인) 클래스가 무엇인지를 제시함.
        counter = Counter(y) # python method의 collection 모듈을 사용해서 개수를 측정하였음.
        most_common = counter.most_common(1)[0][0]
        return most_common

   def predict(self, X): # 현재 제시된 class의 개수와 각 트리의 예측 값을 바탕으로, 가장 설득력 있는 결과는 무엇인지를 제시하는 메소드에 해당함.
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [self._most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)


# 위의 메소드를 사용해서 feature importance를 산출할 경우, 아래의 코드를 사용할 수 있음 - 현재는 계산상의 편의를 위해 sklearn의 rf관련 메소드를 사용함.

# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split

# X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# rf = RandomForest(n_trees=100, max_depth=10)
# rf.fit(X_train, y_train)
# importances = rf.feature_importances(X_train, y_train)

# for i, importance in enumerate(importances):
  #  print(f"Feature {i+1}: {importance}") - 작동 예시