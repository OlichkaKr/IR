import numpy as np

class TwoLayerNet(object):
        

	def __init__(self, input_size, hidden_size, output_size, std):
		self.params = {}
		self.params['W1'] = (((2 / input_size) ** 0.5) *
								 np.random.randn(input_size, hidden_size))
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = (((2 / hidden_size) ** 0.5) *
							   np.random.randn(hidden_size, output_size))
		self.params['b2'] = np.zeros(output_size)

	def loss(self, X, y=None, reg=0.0):
		
		# Unpack variables from the params dictionary
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		N, D = X.shape

		# Compute the forward pass
		l1 = X.dot(W1) + b1
		l1[l1 < 0] = 0
		l2 = l1.dot(W2) + b2
		exp_scores = np.exp(l2)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
		scores = l2

		# Compute the loss

		W1_r = 0.5 * reg * np.sum(W1 * W1)
		W2_r = 0.5 * reg * np.sum(W2 * W2)

		loss = -np.sum(np.log(probs[range(y.shape[0]), y]))/N + W1_r + W2_r


		# Backward pass: compute gradients
		grads = {}
		
		probs[range(X.shape[0]),y] -= 1
		dW2 = np.dot(l1.T, probs)
		dW2 /= X.shape[0]
		dW2 += reg * W2
		grads['W2'] = dW2
		grads['b2'] = np.sum(probs, axis=0, keepdims=True) / X.shape[0]
		
		delta = probs.dot(W2.T)
		delta = delta * (l1 > 0)
		grads['W1'] = np.dot(X.T, delta)/ X.shape[0] + reg * W1
		grads['b1'] = np.sum(delta, axis=0, keepdims=True) / X.shape[0]

		return loss, grads

	def train(self, X, y, X_val, y_val,
		learning_rate=1e-3, learning_rate_decay=0.95,
		reg=5e-6, num_iters=100,
		batch_size=24, verbose=False):
			
		num_train = X.shape[0]
		iterations_per_epoch = max(num_train / batch_size, 1)

		# Use SGD to optimize the parameters in self.model
		loss_history = []
		train_acc_history = []
		val_acc_history = []

		# Training cycle
		for it in range(num_iters):
			# Mini-batch selection
			indexes = np.random.choice(X.shape[0], batch_size,
													 replace=True)
			X_batch = X[indexes]
			y_batch = y[indexes]

			# Compute loss and gradients using the current minibatch
			loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
			loss_history.append(loss)

			# Update weights
			self.params['W1'] -= learning_rate * grads['W1']
			self.params['b1'] -= learning_rate * grads['b1'][0]
			self.params['W2'] -= learning_rate * grads['W2']
			self.params['b2'] -= learning_rate * grads['b2'][0]

			if verbose and it % 100 == 0:
					print('iteration %d / %d: loss %f' % (it,
							num_iters, loss))

			# Every epoch, check accuracy and decay learning rate.
			if it % iterations_per_epoch == 0:
				# Check accuracy
				train_acc = (self.predict(X_batch)==y_batch).mean()
				val_acc = (self.predict(X_val) == y_val).mean()
				train_acc_history.append(train_acc)
				val_acc_history.append(val_acc)

				# Decay learning rate
				learning_rate *= learning_rate_decay

		return {
		  'loss_history': loss_history,
		  'train_acc_history': train_acc_history,
		  'val_acc_history': val_acc_history,
		}

	def predict(self, X):
			
		l1 = X.dot(self.params['W1']) + self.params['b1']
		l1[l1 < 0] = 0
		l2 = l1.dot(self.params['W2']) + self.params['b2']
		exp_scores = np.exp(l2)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
		y_pred = np.argmax(probs, axis=1)

		return y_pred

	def predict_single(self, X):
			
		l1 = X.dot(self.params['W1']) + self.params['b1']
		l1[l1 < 0] = 0
		l2 = l1.dot(self.params['W2']) + self.params['b2']
		exp_scores = np.exp(l2)
		y_pred = np.argmax(exp_scores)
		
		return y_pred