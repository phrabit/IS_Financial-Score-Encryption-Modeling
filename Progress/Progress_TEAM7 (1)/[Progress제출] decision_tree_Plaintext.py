import numpy as np

# 기타 라이브러리를 제외하고, 기본적으로 데이터 처리를 위해 동형암호와 호환하여 사용할 수 있는 numpy 라이브러리 만을 사용하는 것을 목표로 함.
# 이하 코드는 sklearn의 decision tree 관련 코드를 기반으로 하되, 라이브러리를 한정하기 위한 수정(gpt 기반 + 논문 개념 삽입)이 반영되어 있음.
# _(under_bar)의 표현은 기존 라이브러리의 메소드 명을 기반으로 하였음 - 이름에 대한 명명법은 더 파악해야 할 듯?
# 이제 이 친구들을 동형암호로 어떻게 바꿀지에 대해서 생각을 하고 있고, 어느 부분에 대해서 실제로 할 수 있을 것 같다 - 라는 점을 PROGRESS에서 밝히면 될 것.
# 추가적으로 이후 마무리 발표에서는 무엇을 더 하게 될지..까지

class DecisionTree: #decision tree에 관련된 부분
    def __init__(self, max_depth=5, min_samples_split=2): # tree에 대한 default parameter를 입력받는 부분에 해당함
        self.max_depth = max_depth # 깊이
        self.min_samples_split = min_samples_split # 최소 가지의 개수
        self.tree = None # 이번 메소드의 사용을 통해 지정하는 트리 개게

    def fit(self, X, y): # tree의 클래스(분류)와 feature(결정 요소)의 개수를 설정. 이후 데이터를 tree의 각 부분으로 분류함
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def predict(self, X): # tree 객체에 대해서 분류하려는 데이터를 투입하는 메소드(input)
        return [self._predict(inputs) for inputs in X] # 아래의 _predict 메소드로 이어짐.

    def _best_split(self, X, y): # y(분류 클래스)의 개수에 맞춰서 데이터를 최적의 분류로 나누는 메소드
        m = y.size
        if m <= self.min_samples_split:
            return None, None
        # 분류할 수 있는 class의 개수가 parameter의 설정보다 적은 경우 - error 처리와 같다고 보면 됨


        best_score = -1
        best_idx, best_thr = None, None  # thr는 thresholds의 줄임말로, 이 데이터가 어떤 가지로 분류되어야 할지를 말한다.
        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y))) # 이 부분은 라이브러리의 구동 방식을 그대로 가져왔는데, zip 함수의 정확한 역할을 설명할 수가 없음
            num_left = [i for i in range(1, m) if classes[i - 1] != classes[i]] # 구조만 보자면,

            for i in num_left: #
                score = self._information_gain(y, classes[:i], classes[i:]) # 아래의 메소드를 통해서 해당 데이터가 어떤 클래스로 분류되었을 때 최대의 정보 효율을 보이는 지 측정
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_thr = (thresholds[i - 1] + thresholds[i]) / 2
        return best_idx, best_thr
        # 분류할 수 있는 class의 개수가 더 많은 경우

    def _information_gain(self, parent, l_child, r_child): # 각 데이터 별 정보 가중치를 측정함
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self._entropy(parent) - (weight_l * self._entropy(l_child) + weight_r * self._entropy(r_child)) # DM 시간에 배웠던 엔트로피 VALUE 측정 방식과 같다.
        return gain

    def _entropy(self, y):
        proportions = [np.sum(y == c) / len(y) for c in range(self.n_classes_)]
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy
    # 엔트로피 수치 계산에 해당함. 해당 트리의 각 노드(값)이 엔트로피 상으로는 어떠한 값을 가지는 가를 보이고 있음.

    def _grow_tree(self, X, y, depth=0): # 트리 성장에 해당하는 부분 - 새로운 데이터를 넣었을 때, 어떤 식으로 데이터가 연결되는지
        best_idx, best_thr = self._best_split(X, y)
        if depth >= self.max_depth or best_idx is None:
            return np.argmax(np.bincount(y))

        left_idx = X[:, best_idx] < best_thr # 각 트리 훈련 한번 당 트리는 성장하게 된다 - 에 해당하는 부분
        tree = {
            'index': best_idx,
            'threshold': best_thr,
            'left': self._grow_tree(X[left_idx, :], y[left_idx], depth + 1),
            'right': self._grow_tree(X[~left_idx, :], y[~left_idx], depth + 1)
        }
        return tree

    def _predict(self, inputs): # 어떠한 데이터셋을 트리에 넣었을 때, 해당 데이터는 어느 부분으로 분류되는지를 보임
        tree = self.tree
        while True:
            if isinstance(tree, dict): # 트리 TRAINING을 요청시 해당 메소드가 지속적으로 트리를 분류함.
                if inputs[tree['index']] < tree['threshold']:
                    tree = tree['left']
                else:
                    tree = tree['right']
            else:
                return tree