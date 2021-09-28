import torch
import numpy as np
import pandas as pd
import os
import argparse
import random

parser = argparse.ArgumentParser(description="Setup I-CAMF")
parser.add_argument("-t", type=int, default=1, help="Enable training (1 is enabled)")
parser.add_argument("-s", type=int, default=0, help="Save trained features (1 is enabled)")
parser.add_argument("-l", type=int, default=0, help="Load features if available (1 is enabled)")
parser.add_argument("-e", type=int, default=1000, help="Number of epochs")
parser.add_argument("-lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("-f", type=int, default=20, help="Number of features per user/item/context condition")
parser.add_argument("-ld", type=float, default=0.02, help="Regularization term")
parser.add_argument("-os", type=int, default=0, help="Select OS (0 for Windows, 1 for Linux)")

args = parser.parse_args()

if args.os == 0:
	clear = lambda: os.system('cls') # windows
else:
	clear = lambda: os.system('clear') # linux

if args.s == 1 and not os.path.exists("./dat"):
	os.makedirs("./dat")


trips = pd.read_csv("Data_TripAdvisor_v2.csv")

trips.drop(["UserTimeZone", "UserState", "ItemCity", "ItemState", "ItemTimeZone"], axis = 1, inplace = True)

users = trips["UserID"].unique()
items = trips["ItemID"].unique()
contexts = trips["TripType"].unique()

trips = trips.sort_values(by=["UserID"])

loaded = False
if args.l == 1:
	try:
		train_data = np.load("./dat/train_data.npy")
		test_data = np.load("./dat/test_data.npy")
		data = np.load("./dat/data_mat.npy")
		loaded = True
		print("Datasets loaded")
	except:
		pass
if not loaded:
	train_data = np.array([[0,0,0,0]])
	test_data = np.array([[0,0,0,0]])
	data = np.zeros((len(users), len(items), len(contexts)))

	most_items_rated = 0
	big_user = ""

	for u in users:
		rated = trips[trips["UserID"] == u]
		if len(rated) > most_items_rated:
			big_user = u
			most_items_rated = len(rated)

		to_test = len(rated) // 5
		for i in range(to_test):
			n = random.randint(0, len(rated) - 1)
			row = rated.iloc[[n]].to_numpy()
			user, item, context, rating = row[0][0], row[0][1], row[0][3], row[0][2]
			user, item, context = np.where(users==user)[0][0], np.where(items==item)[0][0], np.where(contexts==context)[0][0]
			test_data = np.append(test_data, [[user, item, context, rating]], axis = 0)
			rated = rated.drop([rated.index[n]])

		for _, row in rated.iterrows():
			user, item, context, rating = row["UserID"], row["ItemID"], row["TripType"], row["Rating"]
			user, item, context = np.where(users==user)[0][0], np.where(items==item)[0][0], np.where(contexts==context)[0][0]
			train_data = np.append(train_data, [[user, item, context, rating]], axis = 0)
			data[user][item][context] = rating

	train_data = np.delete(train_data, 0, axis = 0)
	test_data = np.delete(test_data, 0, axis = 0)

	if args.s == 1:
		np.save("./dat/train_data.npy", train_data)
		np.save("./dat/test_data.npy", test_data)
		np.save("./dat/data_mat.npy", data)

	print("User with the most items rated: %d with %d items" % (np.where(users == big_user)[0], most_items_rated))
print("Number of users: %d\nNumber of Items: %d\nNumber of Context Conditions: %d" % (len(users), len(items), len(contexts)))

class MF():
	def __init__(self, data, features, ld, load_files = False):
		self.data = data
		self.ld = ld
		self.features = features
		self.user_count = data.shape[0]
		self.item_count = data.shape[1]
		self.context_count = data.shape[2]
		self.global_mean = np.mean(self.data[np.nonzero(self.data)])
		self.user_features = np.random.uniform(low = 0.1, high = 0.9, size = (self.user_count, self.features))
		self.item_features = np.random.uniform(low = 0.1, high = 0.9, size = (self.item_count, self.features))
		self.context_features = np.random.uniform(low = 0.1, high = 0.9, size = (self.context_count, self.features))
		self.user_bias = np.zeros(self.user_count)
		for u in range(self.user_count):
			self.user_bias[u] = np.mean(data[u][np.nonzero(data[u])]) - self.global_mean
		self.item_bias = np.zeros(self.item_count)
		for i in range(self.item_count):
			self.item_bias[i] = np.mean(data[:,i][np.nonzero(data[:,i])]) - self.global_mean
			if np.isnan(self.item_bias[i]):
				self.item_bias[i] = 0.0
		if load_files:
			try:
				self.user_features = np.load("./dat/user_features.npy")
				self.item_features = np.load("./dat/item_features.npy")
				self.context_features = np.load("./dat/context_features.npy")
				self.user_bias = np.load("./dat/user_bias.npy")
				self.item_bias = np.load("./dat/item_bias.npy")
			except:
				pass

	def pred(self, user, item, context):
		pred = (np.dot(self.user_features[user], self.item_features[item]) 
			+ np.dot(self.user_features[user], self.context_features[context]) 
			+ np.dot(self.item_features[item], self.context_features[context])
			)
		pred += self.global_mean + self.user_bias[user] + self.item_bias[item]
		#pred += self.global_mean + (np.mean(data[user][np.nonzero(data[user])]) - self.global_mean) + (np.mean(data[:,item][np.nonzero(data[:,item])]) - self.global_mean)
		return pred

	def gradient(self, user, item, context, param, r = None):
		u = self.user_features[user]
		i = self.item_features[item]
		c = self.context_features[context]
		b_u = self.user_bias[user]
		b_i = self.user_bias[item]
		err = (r if r != None else self.data[user][item][context]) - self.pred(user = user, item = item, context = context)
		if err > 0:
			x = 1
		else:
			x = -1

		if param == "user":
			gradient = err * (i + c) - self.ld * u
		elif param == "item":
			gradient = err * (u + c) - self.ld * i
		elif param == "context":
			gradient = err * (u + i) - self.ld * c
		elif param == "user_bias":
			gradient = err + self.ld - b_u
		else:
			gradient = err + self.ld - b_i
		return gradient

	def user_feature_gradient(self, user):
		summation_user = 0
		summation_user_bias = 0
		for i in range(0, self.item_count):
			for c in range(0, self.context_count):
				summation_user += self.gradient(user, i, c, "user")
				summation_user_bias += self.gradient(user, i, c, "user_bias")
		return summation_user / (self.item_count * self.context_count), summation_user_bias / (self.item_count * self.context_count)

	def item_feature_gradient(self, item):
		summation_item = 0
		summation_item_bias = 0
		for u in range(0, self.user_count):
			for c in range(0, self.context_count):
				summation_item += self.gradient(u, item, c, "item")
				summation_item_bias += self.gradient(u, item, c, "item_bias")
		return summation_item / (self.user_count * self.context_count), summation_item_bias / (self.user_count * self.context_count)

	def context_feature_gradient(self, context):
		summation = 0
		for u in range(0, self.user_count):
			for i in range(0, self.item_count):
				summation += self.gradient(u, i, context, "context")
		return summation / (self.item_count * self.context_count)

	def train(self, train_data, learning_rate, iterations):
		for n in range(iterations):
			np.random.shuffle(train_data)
			for u, i, c, r in train_data:
				u_grad = self.gradient(u, i, c, "user", r)
				i_grad = self.gradient(u, i, c, "item", r)
				c_grad = self.gradient(u, i, c, "context", r)
				b_u_grad = self.gradient(u, i, c, "user_bias", r)
				b_i_grad = self.gradient(u, i, c, "item_bias", r)
				
				self.user_features[u] += u_grad * learning_rate
				self.item_features[i] += i_grad * learning_rate
				self.context_features[c] += c_grad * learning_rate
				self.user_bias[u] += b_u_grad * learning_rate
				self.item_bias[i] += b_i_grad * learning_rate
			total = 0
			for u, i, c, r in train_data:
				pred = self.pred(u, i, c)
				total += abs(r - pred)
			print("Epoch %d: MAE = %3f" % (n + 1, total / len(train_data)))

		if args.s == 1:
			np.save("./dat/user_features.npy", self.user_features)
			np.save("./dat/item_features.npy", self.item_features)
			np.save("./dat/context_features.npy", self.context_features)
			np.save("./dat/user_bias.npy", self.user_bias)
			np.save("./dat/item_bias.npy", self.item_bias)

R = MF(data, args.f, args.ld, args.l == 1)

if args.t == 1:
	R.train(train_data, args.lr, args.e)

def getPredictions():
	preds = np.zeros((users.shape[0], items.shape[0], contexts.shape[0]))
	for u in range(users.shape[0]):
		for i in range(items.shape[0]):
			for c in range(contexts.shape[0]):
				preds[u][i][c] = R.pred(u, i, c)
	return preds


def getRated(user, from_test = True, from_train = True):
	rated = np.array([[0,0,0,0]])
	if from_train:
		for u, i, c, r in train_data:
			if u == user:
				rated = np.append(rated, [[u,i,c,r]], axis = 0)
	if from_test:
		for u, i, c, r in test_data:
			if u == user:
				rated = np.append(rated, [[u,i,c,r]], axis = 0)
	return np.delete(rated, 0, axis = 0)

def getTopN(preds, N, user, travellers = 0):
	topN = []
	context_filter = []
	context = -1
	#travellers: 0-solo 1-family 2-couples 3-business 4-friends
	if travellers == 1:
		preds = preds[:, [0,3]]
		context_filter = [0,3]
	elif travellers == 2:
		preds = preds[:, 2]
		context = 2
	elif travellers > 2:
		preds = preds[:, [1,4]]
		context_filter = [1,4]
	else:
		rated = getRated(user, False)
		context_list = rated[:,2]
		(values, counts) = np.unique(context_list, return_counts = True)
		ind = np.argmax(counts)
		context = values[ind]
		preds = preds[:, context]

	for _ in range(N):
		item = np.where(preds==np.max(preds))[0]
		try:
			item = item[0]
		except:
			pass
		c = context if context != -1 else context_filter[np.where(preds==np.max(preds))[1][0]]
		topN.append([item, c])
		preds[item] = 0 if context != -1 else np.zeros(preds.shape[1])
	return np.array(topN)

def getConfMat(user, preds, N = 100):
	tp, tn, fp, fn = 0, 0, 0, 0
	topN = getTopN(np.copy(preds), N, user)
	rated = getRated(user, True, False)
	bound = int(round(np.min(topN[:,0])))
	if bound > 5:
		bound = 5
	for u, i, c, r in rated:
		if i in topN[:,0]:
			if r < bound:
				fp += 1
			else:
				tp += 1
		else:
			if r >= bound:
				fn += 1
			else:
				tn += 1
	return np.array([tp, tn, fp, fn])


preds = getPredictions()

total = 0
for u, i, c, r in train_data:
	pred = preds[u][i][c]
	total += abs(r - pred)

tp, tn, fp, fn = 0, 0, 0, 0
for u, i, c, r in test_data:
	pred = R.pred(u, i, c)
	if pred >= 3.5 and r >= 3.5:
		tp += 1
	elif pred >= 3.5 and r < 3.5:
		fp += 1
	elif pred < 3.5 and r >= 3.5:
		fn += 1
	elif pred < 3.5 and r < 3.5:
		tn += 1

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("Accuracy using a rating boundary of 3.5:")
print("Precision: %.3f | Recall: %.3f" % (precision, recall))
print("TP: %d | TN: %d | FP: %d | FN: %d" % (tp, tn, fp, fn))

conf_mat = np.array([0,0,0,0])
for u in range(len(users)):
	conf_mat += getConfMat(u, np.copy(preds[u]))
tp, tn, fp, fn = conf_mat

precision = tp / (tp + fp)
recall = tp / (tp + fn)
print("Accuracy using N = 100:")
print("Precision: %.3f | Recall: %.3f" % (precision, recall))
print("TP: %d | TN: %d | FP: %d | FN: %d" % (tp, tn, fp, fn))

print("\nFinal MAE Loss: %.3f" % (total / len(train_data)))




state = "login"
info = {
	"user": -1,
	"preds": np.zeros((items.shape[0],contexts.shape[0])),
	"travellers": 0,
	"rated": None
}
predicted = False
while True:
	if state == "login":
		try:
			temp = input("Login with userID (0-2370): ")
			if temp == "esc":
				break
			info["user"] = int(temp)
			info["rated"] = getRated(info["user"])
			try:
				clear()
			except:
				pass
			print("\nWelcome, user " + users[info["user"]])
			predicted = False
			state = "menu"
		except:
			print("userID invalid. Try again.\n")
	elif state == "menu":
		if not predicted:
			info["preds"] = preds[info["user"]]
			predicted = True
		try:
			print("\n    \"esc\" to quit")
			print("    \"r N\" to get top N recommendations (0-2268)")
			print("    \"t T\" to set number of travellers to T")
			print("    \"p\" to get predictions of your rated items")

			temp = input("\nInput: ")
			try:
				clear()
			except:
				pass
			if temp == "esc":
				break

			print("\nNumber of Travellers: %s\n" % ("N/A" if info["travellers"] < 1 else info["travellers"]))
			for u, i, c, r in info["rated"]:
				i_rating = np.mean(trips.loc[trips["ItemID"] == items[i]]["Rating"].tolist())
				print("Item: %7s | Your Rating: %d | Overall Rating: %2.1f" % (items[i], int(r), i_rating))
			print("")
			
			if temp[0] == "r":
				N = int(temp[2:])
				topN = getTopN(np.copy(info["preds"]), N, info["user"], info["travellers"])
				for i, c in topN:
					i_rating = np.mean(trips.loc[trips["ItemID"] == items[i]]["Rating"].tolist())
					pred = info["preds"][i][c]
					print("Item: %7s | Predicted Rating: %2.1f | Overall Rating: %2.1f | Context: %s" % (items[i], pred, i_rating, contexts[c]))
			elif temp[0] == "t":
				info["travellers"] = int(temp[2:])
			elif temp[0] == "p":
				for u, i, c, r in info["rated"]:
					pred = info["preds"][i][c]
					print("Item: %7s | Your Rating: %d | Predicted Rating: %2.1f" % (items[i], int(r), pred))
		except:
			print("Input error. Try again.")