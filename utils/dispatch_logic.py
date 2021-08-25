# file name: dispatch_rule.py

import numpy as np

def get_order_from(queue_space, dspch_rule):
	#get oder in queue
	if dspch_rule == 'FIFO':
		order = queue_space[0]
	elif dspch_rule == 'LIFO':
		order = queue_space[-1]
	else:
		if dspch_rule == 'SPT':
			indx  = np.argmin([order.prc_time[order.progress] for order in queue_space])
		elif dspch_rule == 'LPT':
			indx  = np.argmax([order.prc_time[order.progress] for order in queue_space])
		elif dspch_rule == 'LWKR':
			indx  = np.argmin([sum(ord.prc_time[ord.progress:]) for ord in queue_space])
		elif dspch_rule == 'MWKR':
			indx  = np.argmax([sum(ord.prc_time[ord.progress:]) for ord in queue_space])
		elif dspch_rule == 'SSO':
			indx  = np.argmin(_get_subsequence_prc_times(queue_space))
		elif dspch_rule == 'LSO':
			indx  = np.argmax(_get_subsequence_prc_times(queue_space))
		elif dspch_rule == 'SPT+SSO':
			indx  = np.argmin(_get_cur_subsequence_prc_times(queue_space))
		elif dspch_rule == 'LPT+LSO':
			indx  = np.argmax(_get_cur_subsequence_prc_times(queue_space))
		else:
			print(f'[ERROR #1] Here is not a {dspch_rule} rule in set')
			raise NotImplementedError
			# elif self.dspch_rule == 'STPT':
			#     indx  = np.argmin([sum(order.prc_time) for order in self.space])
			# elif self.dspch_rule == 'LTPT':
			#     indx  = np.argmax([sum(order.prc_time) for order in self.space])
			
		order = queue_space[indx]

	return order

def _get_subsequence_prc_times(queue_space):
	lst_sub_prc_times = []
	for order in queue_space:
		if order.progress + 1 < len(order.prc_time):
			lst_sub_prc_times.append(order.prc_time[order.progress + 1])
		else:
			lst_sub_prc_times.append(0)
	return lst_sub_prc_times

def _get_cur_subsequence_prc_times(queue_space):
	lst_res_prc_times = []
	for order in queue_space:
		if order.progress + 2 < len(order.prc_time):
			cur_subsq_times = sum(order.prc_time[order.progress:order.progress + 2])
			lst_res_prc_times.append(cur_subsq_times)
		else:
			cur_subsq_times = sum(order.prc_time[order.progress:])
			lst_res_prc_times.append(cur_subsq_times)
	return lst_res_prc_times