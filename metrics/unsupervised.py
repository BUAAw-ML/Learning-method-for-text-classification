import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

def embed_feature(dataset, pretrained):
	questions = [dataset['words_q'] for i in range(0, len(dataset), 5)]
	features = np.zeros(shape=(len(questions), pretrained.shape[1]))
	for i, question in enumerate(questions):
		for word_id in question:
			features[i] += pretrained[word_id]
		features[i] /= len(question)
	return features


def tfidf_feature(dataset):
	pass


def ger_kmeans(dataset, ini_num):
	avg_embedding = dataset
	estimator = MiniBatchKMeans(ini_num, max_iter=7000)
	estimator.fit(avg_embedding)
	ini_indices = []
	for i in range(ini_num):
		lengths = np.sum((avg_embedding - estimator.cluster_centers_[i]) ** 2, axis=1)
		ini_indices.append(np.argmin(lengths))
	return np.array(ini_indices)


def ger_representive(dataset, pretrained, ini_num, K):
	def insert(order_list, value, is_tuple=True):
		if is_tuple:
			i = 0
			while i < len(order_list):
				if order_list[i][1] < value[1]:
					break
				i += 1
			if len(order_list) == K and i == len(order_list):
				return False
			order_list.insert(i, value)
			return True
		else:
			i = 0
			while i < len(order_list):
				if order_list[i] < value:
					break
				i += 1
			if len(order_list) == K and i == len(order_list):
				return False
			order_list.insert(i, value)
			return True


	avg_embedding = embed_feature(dataset, pretrained)
	# avg_embedding = dataset
	similarity = cosine_similarity(avg_embedding) + 1
	# print('similarity shape:', similarity.shape)
	ini_indices = []
	KSim_set = [[] for i in range(avg_embedding.shape[0])]
	# scores = np.array([np.sum(similarity[i]) for i in range(similarity.shape[0])])
	scores = np.sum(similarity, axis=1)
	first = [True] * avg_embedding.shape[0] # 标记是否每个sample的KSim样本数目第一次达到K
	while len(ini_indices) < ini_num:
		indice = np.argmax(scores)
		scores[indice] = -float('inf')
		ini_indices.append(indice)
		for i in range(len(KSim_set)):
			KSim = KSim_set[i]
			is_successful = insert(KSim, (indice, similarity[indice, i]), is_tuple=True)
			if not is_successful:
				continue
			if first[i] and len(KSim) == K:
				for j in range(len(scores)):
					if j in ini_indices:
						continue
					elif similarity[i, j] <= KSim[K - 1][1]:
						scores[j] -= similarity[i, j]
					else:
						scores[j] -= KSim[K - 1][1]
				first[i] = False
			elif len(KSim) == K + 1:
				for j in range(len(scores)):
					if j in ini_indices or similarity[i, j] <= KSim[K][1]:
						continue
					elif similarity[i, j] <= KSim[K - 1][1]:
						scores[j] -= (similarity[i, j] - KSim[K][1])
					else:
						scores[j] -= (KSim[K - 1][1] - KSim[K][1])
				KSim.pop()
	return np.array(ini_indices)#, scores, first, similarity, KSim_set


def ger_density(dataset, pretrained, ini_num):
	pass


def ger_submodular(dataset, pretrained, ini_num):
	# use submodular function maximization
	def F(Z):
		if len(Z) == 0:
			return 0
		return np.sum(np.max(similarity[:, Z], axis = 1))


	avg_embedding = embed_feature(dataset, pretrained)
	similarity = cosine_similarity(avg_embedding) + 1
	ini_indices = []
	for i in range(ini_num):
		max_gain = 0
		max_indice = -1
		for j in range(avg_embedding.shape[0]):
			if j in ini_indices:
				continue
			gain = F(ini_indices + [j]) - F(ini_indices)
			if gain > max_gain:
				max_gain = gain
				max_indice = j
		assert max_indice != -1
		ini_indices.append(max_indice)
	return np.array(ini_indices)

def ger_submodular(similarity, label, unlabel, uncertainty_sample, sel_num=10):
	# 这个是 name entity recognition中采用的版本，相比较前面的一个版本变为了要标注的样本在已标记的样本上计算收益
	def G(add_ele):
		add_max = similarity[unlabel][:, add_ele]
		return np.sum(np.maximum(add_max - unlabel_max, 0))

	sel_indices = []
	unlabel_max = np.max(similarity[unlabel][:, label], axis=1) # 由于在已标记样本基础上计算收益，所以不初始化为0
	for i in range(sel_num):
		max_gain = 0
		max_indice = -1
		for j in uncertainty_sample:
			if j in sel_indices:
				continue
			gain = G(j)
			if gain > max_gain:
				max_gain = gain
				max_indice = j
		assert max_indice != -1
		add_max = similarity[unlabel][:, max_indice]
		unlabel_max = np.maximum(add_max, unlabel_max)
		sel_indices.append(max_indice)
	return np.array(sel_indices)


#2019-5-10
def ger_submodular(similarity, unlabel, uncertainty_sample, sel_num = 20):
	# similarity: 预先计算的 train_data的pair similarity
	# unlabel: 未标注索引集合
	# uncertainty_sample: 不确定性采样索引集合
	def G(add_ele):
		add_max = similarity[unlabel][:, add_ele]
		return np.sum(np.maximum(add_max - unlabel_max, 0))

	sel_indices = []
	unlabel_max = np.zeros(shape=(len(unlabel)))  # unlabel中每个元素对当前sel_indices中元素的最大similarity值
	for i in range(sel_num):
		max_gain = 0
		max_indice = -1
		for j in uncertainty_sample:
			if j in sel_indices:
				continue
			gain = G(j)
			if gain > max_gain:
				max_gain = gain
				max_indice = j
		# assert max_indice != -1
		if max_indice == -1:
			print('uncertainty_sample number: {}'.format(len(uncertainty_sample)))
			print('i: {}'.format(i))
			print('uncertainty: {}'.format(list(uncertainty_sample)))
			print('sel_indices: {}'.format(sel_indices))
			assert  max_indice != -1
		add_max = similarity[unlabel][:, max_indice]
		unlabel_max = np.maximum(add_max, unlabel_max)
		sel_indices.append(max_indice)
	return np.array(sel_indices)


def ger_submodular_diversity(similarity, unlabel, uncertainty_sample, topic, sel_num = 20, topic_num=3):
    # unlabel uncertainty_sample: 和原来一样
    # topic: len(topic) == len(uncertainty_sample) 是每个uncertainty_sample的topic标签
    # topic_num: 目前应该是3，topic参数中的数字范围应该是[0, topic_num - 1]
    def G(add_ele, ele_topic):
        return np.sqrt(topic_gain[ele_topic] + np.mean(similarity[unlabel][:, add_ele])) - np.sqrt(topic_gain[ele_topic])
    sel_indices = []
    topic_gain = np.zeros(shape=(topic_num))
    for i in range(sel_num):
        max_gain = 0
        max_indice = -1
        max_topic = -1
        # print('select %dth epoch' % i)
        for idx, j in enumerate(uncertainty_sample):
            if j in sel_indices:
                # print('%d gain: -1' % j)
                continue
            gain = G(j, topic[idx])
            # print('%d gain: %f' % (j, gain))
            if gain > max_gain:
                max_gain = gain
                max_indice = j
                max_topic = topic[idx]
        assert max_indice != -1
        # print('%dth epoch select ele: %d' % (i, max_indice))
        sel_indices.append(max_indice)
        topic_gain[max_topic] += np.mean(similarity[unlabel][:, max_indice])
    return np.array(sel_indices)


# 2019-5-24 新的根据对全集的覆盖得到的样本, 其中lamda为超参数需要调整， 需要大于0,
def ger_submodular_cover(similarity, unlabel, uncertainty_sample, lamda=0.25, sel_num=20):
    print("ger_submodular_cover")
    def G(add_ele):
        add_value = similarity[unlabel][:, add_ele]
        return np.sum(np.minimum(current_cover + add_value, unlabel_max)) - current

    unlabel_max = lamda * np.sum(similarity[unlabel][:, uncertainty_sample], axis=1)
    current_cover = np.zeros(shape=(len(unlabel)))
    current = 0
    sel_indices = []
    for i in range(sel_num):
        max_gain = 0
        max_indice = -1
        for j in uncertainty_sample:
            if j in sel_indices:
                continue
            gain = G(j)
            if gain > max_gain:
                max_gain = gain
                max_indice = j
        # assert  max_indice != -1
        sel_indices.append(max_indice)
        current_cover += similarity[unlabel][:, max_indice]
        current = np.sum(np.minimum(current_cover, unlabel_max))
    return np.array(sel_indices)

