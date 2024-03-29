import numpy as np
import tensorflow as tf
import numpy.linalg as la

def weight_variable(shape):
    initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
    return tf.get_variable("weights", shape, initializer=initializer, dtype=tf.float32)


def bias_variable(shape):
    initializer = tf.constant_initializer(0.0)
    return tf.get_variable("biases", shape, initializer=initializer, dtype=tf.float32)

def build_model(input_embedding, label):
	class_num = 3
	with tf.variable_scope('cnn', reuse=tf.AUTO_REUSE):
		with tf.variable_scope('fc1'):
			shape = input_embedding.shape[1]
			W_fc1 = weight_variable([shape, 1024])
			b_fc1 = bias_variable([1024])
			h_fc1 = tf.nn.relu(tf.matmul(input_embedding, W_fc1) + b_fc1)
			# W_fc2 = weight_variable([1024, self.class_num], l2_reg)
		with tf.variable_scope('fc2'):
			W_fc2 = weight_variable([1024, 1024])
			b_fc2 = bias_variable([1024])
			h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
			# W_fc2 = weight_variable([1024, self.class_num], l2_reg)
		with tf.variable_scope('fc3'):
			W_fc3 = weight_variable([1024, class_num])
			b_fc3 = bias_variable([class_num])
			logit = tf.matmul(h_fc2, W_fc3) + b_fc3
	conv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit))
	correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(label, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	return conv_loss,accuracy


def oneHotRepresentation(y, num=3):
    r = []
    for i in range(y.shape[0]):
        l = np.zeros(num)
        l[y[i]] = 1
        r.append(l)
    return np.array(r)

def load_data_snli():
	prefix = '/home/xfdu/bert'
	snli_train_embedding = np.load(prefix+'/NLI Data/snli_1.0/snli_1.0_train.npy')
	snli_dev_embedding = np.load(prefix+'/NLI Data/snli_1.0/snli_1.0_dev.npy')
	snli_train_label = np.load(prefix+'/NLI Data/snli_1.0/snli_1.0_train_label.npy')
	snli_dev_label = np.load(prefix+'/NLI Data/snli_1.0/snli_1.0_dev_label.npy')
	snli_dev_embedding_swapped = np.load(prefix+'/NLI Data/snli_1.0/snli_1.0_dev_swap.npy')
	snli_test_embedding = np.load(prefix+'/NLI Data/snli_1.0/snli_1.0_test.npy')
	snli_test_label = np.load(prefix+'/NLI Data/snli_1.0/snli_1.0_test_label.npy')
	snli_test_embedding_swapped = np.load(prefix+'/NLI Data/snli_1.0/snli_1.0_test_swap.npy')
	

	return snli_train_embedding, snli_train_label, snli_test_embedding, snli_test_embedding_swapped, snli_test_label,\
	snli_dev_embedding, snli_dev_embedding_swapped,snli_dev_label

def load_data_mnli():
	prefix = '/home/xfdu/bert'
	mnli_train_embedding = np.load(prefix+'/NLI Data/multinli_1.0/multinli_1.0_train.npy')
	mnli_test_embedding_matched = np.load(prefix+'/NLI Data/multinli_1.0/multinli_1.0_dev_matched.npy')
	mnli_test_embedding_mismatched = np.load(prefix+'/NLI Data/multinli_1.0/multinli_1.0_dev_mismatched.npy')
	mnli_train_label = np.load(prefix+'/NLI Data/multinli_1.0/multinli_1.0_train_label.npy')
	mnli_test_label_matched = np.load(prefix+'/NLI Data/multinli_1.0/multinli_1.0_dev_matched_label.npy')
	mnli_test_label_mismatched = np.load(prefix+'/NLI Data/multinli_1.0/multinli_1.0_dev_mismatched_label.npy')

	return mnli_train_embedding, mnli_train_label, mnli_test_embedding_matched, mnli_test_label_matched,\
	 mnli_test_embedding_mismatched, mnli_test_label_mismatched


def process_svd_file_mnli(train_data, test_data_matched, test_data_mismatched, k):
	if not os.path.exists('/home/xfdu/bert/NLI Data/multinli_1.0/train_data_svd.npy'):
		train_data = svd_decomposition(train_data, k)
		# print(train_data.shape)
		np.save('/home/xfdu/bert/NLI Data/multinli_1.0/train_data_svd.npy',train_data)
	else:
		train_data = np.load('/home/xfdu/bert/NLI Data/multinli_1.0/train_data_svd.npy')

	if not os.path.exists('/home/xfdu/bert/NLI Data/multinli_1.0/test_data_matched_svd.npy'):
		test_data_matched = svd_decomposition(test_data_matched, k)
		np.save('/home/xfdu/bert/NLI Data/multinli_1.0/test_data_matched_svd.npy',test_data_matched)
	else:
		test_data_matched = np.load('/home/xfdu/bert/NLI Data/multinli_1.0/test_data_matched_svd.npy')

	if not os.path.exists('/home/xfdu/bert/NLI Data/multinli_1.0/test_data_mismatched_svd.npy'):
		test_data_mismatched = svd_decomposition(test_data_mismatched, k)
		np.save('/home/xfdu/bert/NLI Data/multinli_1.0/test_data_mismatched_svd.npy',test_data_mismatched)
	else:
		test_data_mismatched = np.load('/home/xfdu/bert/NLI Data/multinli_1.0/test_data_mismatched_svd.npy')

	return train_data, test_data_matched, test_data_mismatched  

def process_svd_file_snli(train_data, test_data, test_data_swapped, dev_data, dev_data_swapped,k):
	if not os.path.exists('/home/xfdu/bert/NLI Data/snli_1.0/train_data_svd.npy'):
		train_data = svd_decomposition(train_data, k)
		# print(train_data.shape)
		np.save('/home/xfdu/bert/NLI Data/snli_1.0/train_data_svd.npy',train_data)
	else:
		train_data = np.load('/home/xfdu/bert/NLI Data/snli_1.0/train_data_svd.npy')

	if not os.path.exists('/home/xfdu/bert/NLI Data/snli_1.0/test_data_svd.npy'):
		test_data = svd_decomposition(test_data, k)
		np.save('/home/xfdu/bert/NLI Data/snli_1.0/test_data_svd.npy',test_data)
	else:
		test_data = np.load('/home/xfdu/bert/NLI Data/snli_1.0/test_data_svd.npy')


	if not os.path.exists('/home/xfdu/bert/NLI Data/snli_1.0/test_data_swapped_svd.npy'):
		test_data_swapped = svd_decomposition(test_data_swapped, k)
		np.save('/home/xfdu/bert/NLI Data/snli_1.0/test_data_swapped_svd.npy',test_data_swapped)
	else:
		test_data_swapped = np.load('/home/xfdu/bert/NLI Data/snli_1.0/test_data_swapped_svd.npy')

	if not os.path.exists('/home/xfdu/bert/NLI Data/snli_1.0/dev_data_svd.npy'):
		dev_data = svd_decomposition(dev_data, k)
		np.save('/home/xfdu/bert/NLI Data/snli_1.0/dev_data_svd.npy',dev_data)
	else:
		dev_data = np.load('/home/xfdu/bert/NLI Data/snli_1.0/dev_data_svd.npy')

	if not os.path.exists('/home/xfdu/bert/NLI Data/snli_1.0/dev_data_swapped_svd.npy'):
		dev_data_swapped = svd_decomposition(dev_data_swapped, k)
		np.save('/home/xfdu/bert/NLI Data/snli_1.0/dev_data_swapped_svd.npy',dev_data_swapped)
	else:
		dev_data_swapped = np.load('/home/xfdu/bert/NLI Data/snli_1.0/dev_data_swapped_svd.npy')

	return train_data, test_data, test_data_swapped,dev_data,dev_data_swapped  


def process_svd_file_snli_no_save(train_data, test_data, test_data_swapped, dev_data, dev_data_swapped,k):
	train_data = svd_decomposition(train_data, k)
	test_data = svd_decomposition(test_data, k)
	test_data_swapped = svd_decomposition(test_data_swapped, k)
	dev_data = svd_decomposition(dev_data, k)
	dev_data_swapped = svd_decomposition(dev_data_swapped, k)
	return train_data, test_data, test_data_swapped,dev_data,dev_data_swapped  


def process_svd_file_mnli_no_save(train_data, test_data_matched, test_data_mismatched, k):
	train_data = svd_decomposition(train_data, k)
	test_data_matched = svd_decomposition(test_data_matched, k)
	test_data_mismatched = svd_decomposition(test_data_mismatched, k)
	return train_data, test_data_matched, test_data_mismatched  


def train_lr_mnli(args):
	batch_size = args.batch_size
	train_data, train_label, test_data_matched, test_label_matched, test_data_mismatched, test_label_mismatched =\
	load_data_mnli()

	if args.svd:
		train_data, test_data_matched, test_data_mismatched =\
		 process_svd_file_mnli(train_data, test_data_matched, test_data_mismatched, args.k)


	input_embedding = tf.placeholder(tf.float32, (None, train_data.shape[1]))
	y = tf.placeholder(tf.float32, (None, 3))
	loss,acc = build_model(input_embedding, y)
	optimizer = tf.train.AdamOptimizer(1e-4)
	first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "cnn")
	first_train_op = optimizer.minimize(loss, var_list=first_train_vars)
	# import ipdb
	# ipdb.set_trace()
	step = len(train_data) // args.batch_size if len(train_data) % args.batch_size == 0 else len(train_data) // args.batch_size +1

	best_matched = 0
	best_mismatched = 0
	with tf.Session() as sess:
		print('Starting training')
		sess.run(tf.global_variables_initializer())
		# sess.run(tf.initialize_all_variables())        
		for epoch in range(args.epochs):
			loss_epoch = 0
			acc_epoch = 0
			for i in range(step):
				label = oneHotRepresentation(train_label[i*batch_size:i*batch_size+batch_size])
				_, loss_temp,acc_temp = sess.run([first_train_op,loss,acc], feed_dict={input_embedding:train_data[i*batch_size:i*batch_size+batch_size],\
					y:label})
				loss_epoch += loss_temp
				acc_epoch += acc_temp
			print('Epoch:',epoch,'loss is: ', loss_epoch/step, 'acc is: ', acc_epoch/step)

			if epoch % 1 == 0:
				test_matched_step = len(test_data_matched) // args.batch_size if len(test_data_matched) % args.batch_size == 0 \
				else len(test_data_matched) // args.batch_size +1
				loss_epoch = 0
				acc_epoch = 0
				for i in range(test_matched_step):
					label = oneHotRepresentation(test_label_matched[i*batch_size:i*batch_size+batch_size])
					loss_temp,acc_temp = sess.run([loss,acc], feed_dict={input_embedding:test_data_matched[i*batch_size:i*batch_size+batch_size],\
						y:label})
					loss_epoch += loss_temp
					acc_epoch += acc_temp

				if acc_epoch/test_matched_step > best_matched:
					best_matched = acc_epoch/test_matched_step
				print('Epoch:',epoch,'test matched loss is: ', loss_epoch/test_matched_step, \
					'test matched acc is: ', acc_epoch/test_matched_step, 'best is:', best_matched)


				test_mismatched_step = len(test_data_mismatched) // args.batch_size if len(test_data_mismatched) % args.batch_size == 0 \
				else len(test_data_mismatched) // args.batch_size +1
				loss_epoch = 0
				acc_epoch = 0
				for i in range(test_mismatched_step):
					label = oneHotRepresentation(test_label_mismatched[i*batch_size:i*batch_size+batch_size])
					loss_temp,acc_temp = sess.run([loss,acc], feed_dict={input_embedding:test_data_mismatched[i*batch_size:i*batch_size+batch_size],\
						y:label})
					loss_epoch += loss_temp
					acc_epoch += acc_temp

				if acc_epoch/test_mismatched_step > best_mismatched:
					best_mismatched = acc_epoch/test_mismatched_step

				print('Epoch:',epoch,'test mismatched loss is: ', loss_epoch/test_mismatched_step,\
				 'test mismatched acc is: ', acc_epoch/test_mismatched_step, 'best is:', best_mismatched)


			weights = {}
			for v in tf.trainable_variables():
				weights[v.name] = v.eval()

			if args.saveModel == 1:
				saveName = 'mnli_model'
				np.save('weights/' + saveName, weights)

def train_lr_snli(args):
	batch_size = args.batch_size
	train_data, train_label, test_data, test_data_swapped, test_label, dev_data, dev_data_swapped, dev_label =\
	load_data_snli()

	if args.svd:
		train_data, test_data, test_data_swapped, dev_data, dev_data_swapped = process_svd_file_snli(train_data, \
			test_data, test_data_swapped, dev_data, dev_data_swapped,args.k)

	input_embedding = tf.placeholder(tf.float32, (None, train_data.shape[1]))
	y = tf.placeholder(tf.float32, (None, 3))
	loss,acc = build_model(input_embedding, y)
	optimizer = tf.train.AdamOptimizer(1e-4)
	first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "cnn")
	first_train_op = optimizer.minimize(loss, var_list=first_train_vars)

	step = len(train_data) // args.batch_size if len(train_data) % args.batch_size == 0 else len(train_data) // args.batch_size +1

	best_dev = 0
	best_dev_swapped = 0
	best_test = 0
	best_test_swapped = 0

	with tf.Session() as sess:
		print('Starting training')
		sess.run(tf.global_variables_initializer())
		# sess.run(tf.initialize_all_variables())        
		for epoch in range(args.epochs):
			loss_epoch = 0
			acc_epoch = 0
			for i in range(step):
				label = oneHotRepresentation(train_label[i*batch_size:i*batch_size+batch_size])
				_, loss_temp,acc_temp = sess.run([first_train_op,loss,acc], feed_dict={input_embedding:train_data[i*batch_size:i*batch_size+batch_size],\
					y:label})
				loss_epoch += loss_temp
				acc_epoch += acc_temp
			print('Epoch:',epoch,'loss is: ', loss_epoch/step, 'acc is: ', acc_epoch/step)

			if epoch % 1 == 0:
				test_step = len(test_data) // args.batch_size if len(test_data) % args.batch_size == 0 \
				else len(test_data) // args.batch_size +1
				loss_epoch = 0
				acc_epoch = 0
				for i in range(test_step):
					
					label = oneHotRepresentation(test_label[i*batch_size:i*batch_size+batch_size])
					loss_temp,acc_temp = sess.run([loss,acc], feed_dict={input_embedding:test_data[i*batch_size:i*batch_size+batch_size],\
						y:label})
					loss_epoch += loss_temp
					acc_epoch += acc_temp

				if acc_epoch/test_step > best_test:
					best_test = acc_epoch/test_step
				print('Epoch:',epoch,'test loss is: ', loss_epoch/test_step, 'test acc is: ', acc_epoch/test_step, \
					'best is:', best_test)

				test_step = len(test_data_swapped) // args.batch_size if len(test_data_swapped) % args.batch_size == 0 \
				else len(test_data_swapped) // args.batch_size +1
				loss_epoch = 0
				acc_epoch = 0
				for i in range(test_step):
					label = oneHotRepresentation(test_label[i*batch_size:i*batch_size+batch_size])
					loss_temp,acc_temp = sess.run([loss,acc], feed_dict={input_embedding:test_data_swapped[i*batch_size:i*batch_size+batch_size],\
						y:label})
					loss_epoch += loss_temp
					acc_epoch += acc_temp

				if acc_epoch/test_step > best_test_swapped:
					best_test_swapped = acc_epoch/test_step
				print('Epoch:',epoch,'test swapped loss is: ', loss_epoch/test_step, 'test swapped acc is: ', acc_epoch/test_step,\
					'best is: ', best_test_swapped)


				test_step = len(dev_data) // args.batch_size if len(dev_data) % args.batch_size == 0 \
				else len(dev_data) // args.batch_size +1
				loss_epoch = 0
				acc_epoch = 0
				for i in range(test_step):
					
					label = oneHotRepresentation(dev_label[i*batch_size:i*batch_size+batch_size])
					loss_temp,acc_temp = sess.run([loss,acc], feed_dict={input_embedding:dev_data[i*batch_size:i*batch_size+batch_size],\
						y:label})
					loss_epoch += loss_temp
					acc_epoch += acc_temp
				if acc_epoch/test_step > best_dev:
					best_dev = acc_epoch/test_step

				print('Epoch:',epoch,'dev loss is: ', loss_epoch/test_step, 'dev acc is: ', acc_epoch/test_step,\
					'best is:', best_dev)

				test_step = len(dev_data_swapped) // args.batch_size if len(dev_data_swapped) % args.batch_size == 0 \
				else len(dev_data_swapped) // args.batch_size +1
				loss_epoch = 0
				acc_epoch = 0
				for i in range(test_step):
					label = oneHotRepresentation(dev_label[i*batch_size:i*batch_size+batch_size])
					loss_temp,acc_temp = sess.run([loss,acc], feed_dict={input_embedding:dev_data_swapped[i*batch_size:i*batch_size+batch_size],\
						y:label})
					loss_epoch += loss_temp
					acc_epoch += acc_temp
				if acc_epoch/test_step > best_dev_swapped:
					best_dev_swapped = acc_epoch/test_step

				print('Epoch:',epoch,'dev swapped loss is: ', loss_epoch/test_step, 'dev swapped acc is: ', acc_epoch/test_step,\
					'best is:', best_dev_swapped)


			weights = {}
			for v in tf.trainable_variables():
				weights[v.name] = v.eval()

			if args.saveModel == 1:
				saveName = 'snli_model'
				np.save('weights/' + saveName, weights)

def save_prediction(prediction, name):
	np.save('./'+name+'.npy',prediction)


def lr_snli(args):
	from sklearn.linear_model import LogisticRegression
	train_data, train_label, test_data, test_data_swapped, test_label, dev_data, dev_data_swapped, dev_label =\
	load_data_snli()

	train_data_mnli, train_label_mnli, test_data_matched_mnli, test_label_matched_mnli, \
	test_data_mismatched_mnli, test_label_mismatched_mnli =\
	load_data_mnli()

	if args.svd:
		train_data, test_data, test_data_swapped, dev_data, dev_data_swapped = process_svd_file_snli_no_save(train_data, \
			test_data, test_data_swapped, dev_data, dev_data_swapped,args.k)
		train_data_mnli, test_data_matched_mnli, test_data_mismatched_mnli =\
		 process_svd_file_mnli_no_save(train_data_mnli, test_data_matched_mnli, test_data_mismatched_mnli, args.k)

	model = LogisticRegression()

	model.fit(train_data,train_label)
	test_predictions = model.score(test_data,test_label)
	dev_predictions = model.score(dev_data, dev_label)
	test_swapped_predictions = model.score(test_data_swapped,test_label)
	dev_swapped_predictions = model.score(dev_data_swapped,dev_label)
	#cross domain acc
	dev_matched_predictions = model.score(test_data_matched_mnli, test_label_matched_mnli)
	dev_mismatched_predictions = model.score(test_data_mismatched_mnli,test_label_mismatched_mnli)

	matched_class=model.predict(test_data_matched_mnli)
	mismatched_class=model.predict(test_data_mismatched_mnli)
	if not args.svd:
		np.save('mnli_matched.npy',matched_class)
		np.save('mnli_mismatched.npy',mismatched_class)
	else:
		np.save('mnli_matched_svd.npy',matched_class)
		np.save('mnli_mismatched_svd.npy',mismatched_class)


	if args.svd:
		print('using svd!')
	from joblib import dump, load
	dump(model, 'snli_'+str(args.svd)+'_'+str(args.k)+'.joblib')
	# clf = load('filename.joblib')  
	print('source domain snli')
	print(test_predictions, dev_predictions, test_swapped_predictions, dev_swapped_predictions)
	print('cross domain mnli')
	print(dev_matched_predictions,dev_mismatched_predictions)



def lr_mnli(args):
	from sklearn.linear_model import LogisticRegression
	train_data, train_label, test_data_matched, test_label_matched, test_data_mismatched, test_label_mismatched =\
	load_data_mnli()

	train_data_snli, train_label_snli, test_data_snli, test_data_swapped_snli, test_label_snli,\
	 dev_data_snli, dev_data_swapped_snli, dev_label_snli =\
	load_data_snli()


	if args.svd:
		train_data, test_data_matched, test_data_mismatched =\
		 process_svd_file_mnli_no_save(train_data, test_data_matched, test_data_mismatched, args.k)
		train_data_snli, test_data_snli, test_data_swapped_snli, dev_data_snli, dev_data_swapped_snli\
		 = process_svd_file_snli_no_save(train_data_snli, \
			test_data_snli, test_data_swapped_snli, dev_data_snli, dev_data_swapped_snli,args.k)


	model = LogisticRegression()
	model.fit(train_data,train_label)
	dev_matched_predictions = model.score(test_data_matched, test_label_matched)
	dev_mismatched_predictions = model.score(test_data_mismatched,test_label_mismatched)
	#cross domain
	test_predictions = model.score(test_data_snli,test_label_snli)
	dev_predictions = model.score(dev_data_snli, dev_label_snli)
	test_swapped_predictions = model.score(test_data_swapped_snli,test_label_snli)
	dev_swapped_predictions = model.score(dev_data_swapped_snli,dev_label_snli)



	test_class=model.predict(test_data_snli)
	dev_class=model.predict(dev_data_snli)
	test_swapped_class=model.predict(test_data_swapped_snli)
	dev_swapped_class=model.predict(dev_data_swapped_snli)
	if not args.svd:
		np.save('snli_test.npy',test_class)
		np.save('snli_dev.npy',dev_class)
		np.save('snli_test_swapped.npy',test_swapped_class)
		np.save('snli_dev_swapped.npy',dev_swapped_class)
	else:
		np.save('snli_test_svd.npy',test_class)
		np.save('snli_dev_svd.npy',dev_class)
		np.save('snli_test_swapped_svd.npy',test_swapped_class)
		np.save('snli_dev_swapped_svd.npy',dev_swapped_class)

	if args.svd:
		print('using svd!')
	from joblib import dump, load
	dump(model, 'mnli_'+str(args.svd)+'_'+str(args.k)+'.joblib')
	# clf = load('filename.joblib')  
	print('source domain mnli')
	print(dev_matched_predictions, dev_mismatched_predictions)
	print('cross domain snli')
	print(test_predictions, dev_predictions, test_swapped_predictions, dev_swapped_predictions)

def svd_decomposition(data, k):

	# singular value decomposition
	U, s, V = la.svd(data, full_matrices=False)
	# choose top k important singular values (or eigens)
	Uk = U[:, 1:k]
	Sk = np.diag(s[1:k])
	Vk = V[1:k, :]
	# import ipdb
	# ipdb.set_trace()
	# plot_eigen_values(s[:10])
	# recover 
	data_new = np.matmul(Uk, np.matmul(Sk ,Vk))

	# import ipdb
	# ipdb.set_trace()
	# import matplotlib.pyplot as plt
	# plt.plot(np.sort((data_new-data).reshape(-1))[:-100])
	# plt.show()

	return data_new

def plot_eigen_values(data):
	import matplotlib.pyplot as plt
	plt.plot(data,'*')
	plt.show()

def svd_decomposition_single(data, k):
	recovered_data = []
	iterations=0
	for sub in data:
		iterations +=1 
		# singular value decomposition
		U, s, V = la.svd(sub.reshape(1,sub.shape[0]), full_matrices=False)
		# choose top k important singular values (or eigens)
		Uk = U[:, 0:k]
		Sk = np.diag(s[0:k])
		Vk = V[0:k, :]
		# recover 
		data_new = Uk * Sk * Vk
		data_new = data_new.reshape(-1)
		assert data_new.shape[0] == sub.shape[0]
		if iterations % 2000 == 0:
			print('done more 200 data!')
		recovered_data.append(data_new)

	return np.asarray(recovered_data)

def seed_everything(seed):
    import random
    import numpy as np
    random.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

def differ_index_function(prediction, prediction_svd, label):#differ_index_function_while_svd_is_true
	svd_true_index=[idx for idx in range(len(prediction_svd)) if prediction_svd[idx]==label[idx]]
	differ_index=[idx for idx in svd_true_index if prediction_svd[idx]!=prediction[idx]]

	return differ_index

# def differ_index_function(prediction, prediction_svd, label):
# 	# svd_true_index=[idx for idx in range(len(prediction_svd)) if prediction_svd[idx]==label[idx]]
# 	differ_index=[idx for idx in range(len(prediction_svd)) if prediction_svd[idx]!=prediction[idx]]

# 	return differ_index


def extract_different_prediction_snli2mnli():
	mismatched_label=np.load('/home/xfdu/bert/NLI Data/multinli_1.0/multinli_1.0_dev_mismatched_label.npy')
	matched_label=np.load('/home/xfdu/bert/NLI Data/multinli_1.0/multinli_1.0_dev_matched_label.npy')
	prediction_matched=np.load('./mnli_matched.npy')
	prediction_mismatched=np.load('./mnli_mismatched.npy')
	prediction_matched_svd=np.load('./mnli_matched_svd.npy')
	prediction_mismatched_svd=np.load('./mnli_mismatched_svd.npy')

	matched_differ_index = differ_index_function(prediction_matched,prediction_matched_svd, matched_label)
	mismatched_differ_index=differ_index_function(prediction_mismatched,prediction_mismatched_svd,mismatched_label)
	with open('/home/xfdu/bert/NLI Data/multinli_1.0/multinli_1.0_dev_matched_input.txt') as foo:
		matched_text=foo.readlines()
	with open('/home/xfdu/bert/NLI Data/multinli_1.0/multinli_1.0_dev_mismatched_input.txt') as foo:
		mismatched_text=foo.readlines()


	file=open('./cross_domain_snli2mnli_different_predictions_dev_matched_2.txt', 'w')
	for index in matched_differ_index:
		file.write(matched_text[index])
	file.close()
	file=open('./cross_domain_snli2mnli_different_predictions_dev_mismatched_2.txt', 'w')
	for index in mismatched_differ_index:
		file.write(mismatched_text[index])
	file.close()

			
	return matched_text, mismatched_text, matched_differ_index, mismatched_differ_index

def extract_different_prediction_mnli2snli():
	dev_label=np.load('/home/xfdu/bert/NLI Data/snli_1.0/snli_1.0_dev_label.npy')
	test_label=np.load('/home/xfdu/bert/NLI Data/snli_1.0/snli_1.0_test_label.npy')
	prediction_dev=np.load('./snli_dev.npy')
	prediction_dev_svd=np.load('./snli_dev_svd.npy')
	prediction_test=np.load('./snli_test.npy')
	prediction_test_svd=np.load('./snli_test_svd.npy')
	prediction_dev_swapped=np.load('./snli_dev_swapped.npy')
	prediction_dev_swapped_svd=np.load('./snli_dev_swapped_svd.npy')
	prediction_test_swapped=np.load('./snli_test_swapped.npy')
	prediction_test_swapped_svd=np.load('./snli_test_swapped_svd.npy')

	dev_differ_index = differ_index_function(prediction_dev,prediction_dev_svd, dev_label)
	dev_swapped_differ_index=differ_index_function(prediction_dev_swapped,prediction_dev_swapped_svd,dev_label)
	test_differ_index = differ_index_function(prediction_test,prediction_test_svd, test_label)
	test_swapped_differ_index=differ_index_function(prediction_test_swapped,prediction_test_swapped_svd,test_label)
	with open('/home/xfdu/bert/NLI Data/snli_1.0/snli_1.0_dev_input.txt') as foo:
		dev_text=foo.readlines()
	with open('/home/xfdu/bert/NLI Data/snli_1.0/snli_1.0_dev_swap_input.txt') as foo:
		dev_swapped_text=foo.readlines()
	with open('/home/xfdu/bert/NLI Data/snli_1.0/snli_1.0_test_input.txt') as foo:
		test_text=foo.readlines()
	with open('/home/xfdu/bert/NLI Data/snli_1.0/snli_1.0_test_swap_input.txt') as foo:
		test_swapped_text=foo.readlines()	

	file=open('./cross_domain_mnli2snli_different_predictions_dev_2.txt', 'w')
	for index in dev_differ_index:
		file.write(dev_text[index])
	file.close()
	file=open('./cross_domain_mnli2snli_different_predictions_test_2.txt', 'w')
	for index in test_differ_index:
		file.write(test_text[index])
	file.close()


	return dev_text,dev_swapped_text,test_text,test_swapped_text,\
	 dev_differ_index, dev_swapped_differ_index, test_differ_index, test_swapped_differ_index

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=100, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=100, help='Batch size during training per GPU')
    parser.add_argument('-x', '--training', type=int, default=1, help='train or inference')
    parser.add_argument('-g', '--gpu', type=int, default=1, help='which GPU to use')
    parser.add_argument('-seed', '--seed', type=int, default=100, help='seed')
    parser.add_argument('-s', '--saveModel', type=int, default=1, help='Whether we save this model')
    parser.add_argument('-a', '--action', type=int, default=0, help='which action to work on')
    parser.add_argument('-svd', '--svd', type=int, default=1, help='whether to use svd to denoise the data.')
    parser.add_argument('-k', '--k', type=int, default=768, help='the number of the singular values to keep.')

    args = parser.parse_args()
    seed_everything(args.seed)
    # np.random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    if args.action == 0:
        lr_mnli(args)
    if args.action == 1:
        lr_snli(args)
    if args.action == 2:
    	matched_text, mismatched_text, matched_differ_index, mismatched_differ_index=extract_different_prediction_snli2mnli()
    	import ipdb
    	ipdb.set_trace()
    if args.action == 3:
    	dev_text,dev_swapped_text,test_text,test_swapped_text,\
    	 dev_differ_index, dev_swapped_differ_index, test_differ_index, test_swapped_differ_index = \
    	 extract_different_prediction_mnli2snli()
    	import ipdb
    	ipdb.set_trace()



