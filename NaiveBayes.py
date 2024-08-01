from utils import *
import heapq

class Dataset:
	"""
	A data set for a machine learning problem. It has the following fields:

	d.examples	 A list of training examples. Each one contains attributes
	d.target	 Return the target attribute. By default the final attribute
	d.attrs 	 A list of integer to index into an example. Normally The 
				 same as range(len(d.examples[0]))
	d.attr_names Optional list of names of attributes.
	d.inputs	 A list of attributes without the target.
	d.values	 A list of list: each sublist contains values of the 
				 corresponding attribute. If initially None, it is computed 
				 from known examples by self.set_problem. If not None, an 
				 erroneous value raises ValueError.
	d.distance	 A function to calculate distance between examples.
	d.name 		 Name of the dataset
	d.source 	 URL or other source that the data came from.
	d.exclude	 A list of input attributes that should be excluded as not
				 used. Elements of this list can either be integers or attr_names.

	Normally, you call the constructor and you're done; then you call fields to 
	work.
	"""

	def __init__(self, examples=None, target=-1, attrs=None, attr_names=None, inputs=None, values=None,
				 distance=None, name='', source='', exclude=()):
		"""
		Accept any fields of the dataset. Examples can be a string 
		or csv file from which to parse examples using parse_csv. 
		Optional parameter: exclude, as documented in .set_problem()
		"""
		self.name = name 
		self.distance = distance
		self.source = source
		self.values = values
		self.got_values_flag = bool(values)

		# Initialize examples from string or file or list
		if isinstance(examples, str):
			self.examples = parse_csv(examples)
		elif examples is None:
			self.examples = parse_csv(open_data(name + '.csv').read())
		else:
			self.examples = examples

		# attrs are the indices of examples. Unless otherwise stated.
		if self.examples is not None and attrs is None:
			attrs = list(range(len(self.examples[0])))
		self.attrs = attrs

		# attribute names can come from list, string.
		if isinstance(attr_names, str):
			self.attr_names = attr_names.split()
		else:
			self.attr_names = attr_names or attrs
		self.set_problem(target, inputs=inputs, exclude=exclude)

	def set_problem(self, target, inputs=None, exclude=()):
		"""
		Set or change target and/or input. In this way, one dataset can be 
		used multiple ways. Input, if specified, self.input is a list of
		attributes, or specify exclude as a list of attributes not to use
		in inputs. Attributes can be integer or attr_names. Also compute 
		the values if not given.
		"""

		self.target = self.attr_num(target)
		exclude = list(map(self.attr_num, exclude))
		if inputs:
			self.inputs = remove_all(self.target, inputs)
		else:
			self.inputs = [a for a in self.attrs if a != self.target and a not in exclude]
		if not self.values:
			self.update_values()
		self.check_me()

	def check_me(self):
		assert len(self.attrs) == len(self.attr_names)
		assert set(self.inputs).issubset(set(self.attrs))
		assert self.target not in self.inputs
		assert self.target in self.attrs
		if self.got_values_flag:
			# Only check if values are initialized while intializing dataset
			list(map(self.check_examples, self.examples))

	def add_examples(self, example):
		"""Add an example to the dataset, checking it first"""
		self.check_examples(example)
		self.examples.append(example)

	def check_examples(self, example):
		"""Raise ValueError if example contains an invalid value"""
		if self.values:
			for a in self.attrs:
				if example[a] not in self.values[a]:
					raise ValueError('Bad value {} for attribute {} in {}'
                                     .format(example[a], self.attr_names[a], example))

	def update_values(self):
		# Return a set of values of attributes
		self.values = list(map(unique, zip(*self.examples)))

	def attr_num(self, attr):
		if isinstance(attr, str):
			return self.attr_names.index(attr)
		elif attr < 0:
			return len(self.attrs) + attr
		else:
			return attr  

class CountingProbDist:
	def __init__(self, observation=[], default=0):
		self.dictionary = {}
		self.n_obs = 0
		self.default = 0

		for o in observation:
			self.add(o)

	def smooth_for(self, o):
		"""
		Include o among observations, whether or not it's been observered yet
		"""
		if o not in self.dictionary:
			self.dictionary[o] = self.default
			self.n_obs += self.default

	def add(self, o):
		self.smooth(o)
		self.dictionary[o] += 1
		self.n_obs += 1

	def top(self, n):
		"""Return (obs, count) tuples for the n observations that have the most frequency"""
		return heapq.nlargest(n, [(v, c) for (v, c) in self.dictionary.item()])

	def __getitem__(self, item):
		"""Return a probability for an item"""
		self.smooth(item)
		return self.dictionary[item] / self.n_obs

def NaiveBayesLearner(dataset):
	target_values = dataset.values[dataset.target]
	target_dist = CountingProbDist(target_values)
	attr_dist = {(attr, class_val): CountingProbDist(dataset.values[attr]) for attr in dataset.inputs for class_val in target_values}
	for example in dataset.examples:
		target_val = example[dataset.target]
		target_dist.add(target_val)
		for attr in dataset.inputs:
			attr_dist[attr, target_val].add(example[attr])

	def predict(sample):
		def class_probability(target_val):
			return target_dist[target_val]*product(attr_dist[attr, target_val][sample[attr]] for attr in dataset.inputs)
		return max(target_values, key=class_probability)

	return predict