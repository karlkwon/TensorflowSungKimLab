{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "https://github.com/nlintz/TensorFlow-Tutorials"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "# XOR\r\n",
      "# x1\tx2\ty\r\n",
      "0\t0\t0\r\n",
      "0\t1\t1\r\n",
      "1\t0\t1\r\n",
      "1\t1\t0\r\n"
     ]
    }
   ],
   "source": [
    "!cat /data/hunkim/data_labs/lab_09/train.txt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "a\n"
     ]
    }
   ],
   "source": [
    "from __future__ import print_function\n",
    "print(\"a\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0.  0.  1.  1.]\n",
      " [ 0.  1.  0.  1.]]\n",
      "[ 0.  1.  1.  1.]\n",
      "0 0.733488 [[-0.66880959  0.60531205]]\n",
      "500 0.417903 [[ 0.22604316  1.20360899]]\n",
      "1000 0.350564 [[ 0.70788294  1.50376427]]\n",
      "1500 0.30584 [[ 1.07674897  1.73948407]]\n",
      "2000 0.270902 [[ 1.38974833  1.94790316]]\n",
      "2500 0.242684 [[ 1.66568935  2.13963127]]\n",
      "3000 0.219478 [[ 1.91335893  2.3187263 ]]\n",
      "3500 0.200102 [[ 2.13805056  2.48717284]]\n",
      "4000 0.183704 [[ 2.34348965  2.64618921]]\n",
      "4500 0.169659 [[ 2.53253198  2.79667163]]\n",
      "5000 0.157503 [[ 2.70744586  2.93935442]]\n",
      "5500 0.146884 [[ 2.87007904  3.07487535]]\n",
      "6000 0.137535 [[ 3.02195764  3.20380187]]\n",
      "6500 0.129243 [[ 3.16434979  3.32664514]]\n",
      "7000 0.121843 [[ 3.29832053  3.44386744]]\n",
      "7500 0.115201 [[ 3.42477345  3.55588818]]\n",
      "8000 0.109209 [[ 3.54447532  3.66308761]]\n",
      "8500 0.103778 [[ 3.65808749  3.76581144]]\n",
      "9000 0.0988349 [[ 3.76617908  3.86437535]]\n",
      "9500 0.0943182 [[ 3.86924624  3.95906639]]\n",
      "10000 0.0901763 [[ 3.96771979  4.05014372]]\n",
      "[array([[ 0.18780643,  0.92994314,  0.92437941,  0.998577  ]], dtype=float32), array([[ 0.,  1.,  1.,  1.]], dtype=float32), array([[ True,  True,  True,  True]], dtype=bool), 1.0]\n",
      "1.0\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "\n",
    "xy = np.loadtxt('/data/hunkim/data_labs/lab_09/train.txt', unpack=True)\n",
    "# x_data = np.transpose(xy[0:-1])\n",
    "# y_data = np.transpose(xy[-1])\n",
    "x_data = (xy[0:-1])\n",
    "y_data = (xy[-1])\n",
    "\n",
    "y_data[3] = 1.\n",
    "print(x_data)\n",
    "print(y_data)\n",
    "\n",
    "X = tf.placeholder(tf.float32)\n",
    "Y = tf.placeholder(tf.float32)\n",
    "\n",
    "W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))\n",
    "b = tf.Variable(tf.zeros([1]), name=\"Bias2\")\n",
    "\n",
    "h = tf.matmul(W, X) + b\n",
    "# hypothesis = tf.div(1., 1. + tf.exp(-h))\n",
    "hypothesis = tf.sigmoid(h)\n",
    "cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1 - hypothesis))\n",
    "\n",
    "a = tf.Variable(0.01)\n",
    "optimizer = tf.train.GradientDescentOptimizer(a)\n",
    "train = optimizer.minimize(cost)\n",
    "\n",
    "init = tf.initialize_all_variables()\n",
    "\n",
    "with tf.Session() as sess:\n",
    "    sess.run(init)\n",
    "    \n",
    "    for step in xrange(10001):\n",
    "        sess.run(train, feed_dict={X:x_data, Y:y_data})\n",
    "        if step % 500 == 0:\n",
    "            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))\n",
    "            \n",
    "    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)\n",
    "    accuracy = tf.reduce_mean(tf.cast(correct_prediction, \"float\"))\n",
    "    \n",
    "    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data}))\n",
    "\n",
    "#     print \"Accuracy: \" + accuracy.eval({X:x_data, Y:y_data})\n",
    "    print(accuracy.eval({X:x_data, Y:y_data}))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(2, 4)\n",
      "2\n"
     ]
    }
   ],
   "source": [
    "print(x_data.shape)\n",
    "print(len(x_data))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.,  0.,  1.,  1.],\n",
       "       [ 0.,  1.,  0.,  1.],\n",
       "       [ 0.,  1.,  1.,  0.]])"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "xy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(2, 4) (4,)\n",
      "0 0.729931\n",
      "1000 0.692951\n",
      "2000 0.692522\n",
      "3000 0.691845\n",
      "4000 0.690282\n",
      "5000 0.685532\n",
      "6000 0.6679\n",
      "7000 0.607883\n",
      "8000 0.451525\n",
      "9000 0.240745\n",
      "10000 0.1339\n",
      "11000 0.0871604\n",
      "12000 0.0631522\n",
      "13000 0.048997\n",
      "14000 0.039801\n",
      "15000 0.0333992\n",
      "16000 0.0287094\n",
      "17000 0.0251375\n",
      "18000 0.0223327\n",
      "19000 0.0200754\n",
      "[array([[ 0.021301  ,  0.98395592,  0.98337168,  0.01825237]], dtype=float32), array([[ 0.,  1.,  1.,  0.]], dtype=float32), array([[ True,  True,  True,  True]], dtype=bool), 1.0]\n",
      "1.0\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "\n",
    "xy = np.loadtxt('/data/hunkim/data_labs/lab_09/train.txt', unpack=True)\n",
    "# x_data = np.transpose(xy[0:-1])\n",
    "# y_data = np.transpose(xy[-1])\n",
    "x_data = xy[0:-1]\n",
    "y_data = xy[-1]\n",
    "\n",
    "print x_data.shape, y_data.shape\n",
    "\n",
    "X = tf.placeholder(tf.float32, name = 'X-input')\n",
    "Y = tf.placeholder(tf.float32, name = 'Y-input')\n",
    "\n",
    "# 2 - input, 2 - output\n",
    "W1 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name = \"Weight1\")\n",
    "# 2 - input, 1 - output\n",
    "W2 = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name = \"Weight2\")\n",
    "\n",
    "b1 = tf.Variable(tf.zeros([2, 1]), name=\"Bias1\")\n",
    "b2 = tf.Variable(tf.zeros([1]), name=\"Bias2\")\n",
    "\n",
    "L2 = tf.sigmoid(tf.matmul(W1, X) + b1)\n",
    "hypothesis = tf.sigmoid(tf.matmul(W2, L2) + b2)\n",
    "# hypothesis = tf.sigmoid(tf.matmul(W2, X) + b2)\n",
    "\n",
    "cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1 - hypothesis))\n",
    "\n",
    "a = tf.Variable(0.05)\n",
    "optimizer = tf.train.GradientDescentOptimizer(a)\n",
    "train = optimizer.minimize(cost)\n",
    "\n",
    "init = tf.initialize_all_variables()\n",
    "\n",
    "with tf.Session() as sess:\n",
    "    sess.run(init)\n",
    "    \n",
    "    merged = tf.merge_all_summaries()\n",
    "    writer = tf.train.SummaryWriter(\"./logs/xor_logs\", sess.graph_def)\n",
    "    \n",
    "    for step in xrange(20000):\n",
    "        sess.run(train, feed_dict={X:x_data, Y:y_data})\n",
    "        if step % 1000 == 0:\n",
    "#             summary = sess.run(merged, feed_dict={X:x_data, Y:y_data})\n",
    "#             writer.add_summary(summary, step)\n",
    "#             print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W1), sess.run(W2)\n",
    "            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data})\n",
    "            \n",
    "    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)\n",
    "    accuracy = tf.reduce_mean(tf.cast(correct_prediction, \"float\"))\n",
    "    \n",
    "    print sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data})\n",
    "\n",
    "#     print \"Accuracy: \" + accuracy.eval({X:x_data, Y:y_data})\n",
    "    print accuracy.eval({X:x_data, Y:y_data})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(2, 4) (4,)\n",
      "0 0.774346\n",
      "1000 0.677561\n",
      "2000 0.641134\n",
      "3000 0.559344\n",
      "4000 0.433221\n",
      "5000 0.285784\n",
      "6000 0.169868\n",
      "7000 0.103616\n",
      "8000 0.0685228\n",
      "9000 0.0489105\n",
      "10000 0.0370437\n",
      "11000 0.0293308\n",
      "12000 0.0240173\n",
      "13000 0.0201823\n",
      "14000 0.0173091\n",
      "15000 0.0150903\n",
      "16000 0.0133334\n",
      "17000 0.0119131\n",
      "18000 0.0107443\n",
      "19000 0.00976813\n",
      "[array([[ 0.00454903,  0.9910506 ,  0.99082762,  0.01292328]], dtype=float32), array([[ 0.,  1.,  1.,  0.]], dtype=float32), array([[ True,  True,  True,  True]], dtype=bool), 1.0]\n",
      "1.0\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "\n",
    "xy = np.loadtxt('/data/hunkim/data_labs/lab_09/train.txt', unpack=True)\n",
    "# x_data = np.transpose(xy[0:-1])\n",
    "# y_data = np.transpose(xy[-1])\n",
    "x_data = xy[0:-1]\n",
    "y_data = xy[-1]\n",
    "\n",
    "print x_data.shape, y_data.shape\n",
    "\n",
    "X = tf.placeholder(tf.float32, name = 'X-input')\n",
    "Y = tf.placeholder(tf.float32, name = 'Y-input')\n",
    "\n",
    "# 2 - input, 2 - output\n",
    "W1 = tf.Variable(tf.random_uniform([10, 2], -1.0, 1.0), name = \"Weight1\")\n",
    "# 2 - input, 1 - output\n",
    "W2 = tf.Variable(tf.random_uniform([1, 10], -1.0, 1.0), name = \"Weight2\")\n",
    "\n",
    "b1 = tf.Variable(tf.zeros([10, 1]), name=\"Bias1\")\n",
    "b2 = tf.Variable(tf.zeros([1]), name=\"Bias2\")\n",
    "\n",
    "L2 = tf.sigmoid(tf.matmul(W1, X) + b1)\n",
    "hypothesis = tf.sigmoid(tf.matmul(W2, L2) + b2)\n",
    "# hypothesis = tf.sigmoid(tf.matmul(W2, X) + b2)\n",
    "\n",
    "cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1 - hypothesis))\n",
    "\n",
    "a = tf.Variable(0.05)\n",
    "optimizer = tf.train.GradientDescentOptimizer(a)\n",
    "train = optimizer.minimize(cost)\n",
    "\n",
    "init = tf.initialize_all_variables()\n",
    "\n",
    "with tf.Session() as sess:\n",
    "    sess.run(init)\n",
    "    \n",
    "    merged = tf.merge_all_summaries()\n",
    "    writer = tf.train.SummaryWriter(\"./logs/xor_logs\", sess.graph_def)\n",
    "    \n",
    "    for step in xrange(20000):\n",
    "        sess.run(train, feed_dict={X:x_data, Y:y_data})\n",
    "        if step % 1000 == 0:\n",
    "#             summary = sess.run(merged, feed_dict={X:x_data, Y:y_data})\n",
    "#             writer.add_summary(summary, step)\n",
    "#             print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W1), sess.run(W2)\n",
    "            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data})\n",
    "            \n",
    "    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)\n",
    "    accuracy = tf.reduce_mean(tf.cast(correct_prediction, \"float\"))\n",
    "    \n",
    "    print sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data})\n",
    "\n",
    "#     print \"Accuracy: \" + accuracy.eval({X:x_data, Y:y_data})\n",
    "    print accuracy.eval({X:x_data, Y:y_data})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(2, 4) (4,)\n",
      "0 0.711848\n",
      "1000 0.693038\n",
      "2000 0.692576\n",
      "3000 0.691849\n",
      "4000 0.690465\n",
      "5000 0.687568\n",
      "6000 0.68105\n",
      "7000 0.664195\n",
      "8000 0.615849\n",
      "9000 0.518371\n",
      "10000 0.323382\n",
      "11000 0.108966\n",
      "12000 0.0461885\n",
      "13000 0.0262391\n",
      "14000 0.0175404\n",
      "15000 0.0128918\n",
      "16000 0.0100658\n",
      "17000 0.00819194\n",
      "18000 0.00686995\n",
      "19000 0.00589306\n",
      "[array([[ 0.00294126,  0.99464744,  0.99433881,  0.00657131]], dtype=float32), array([[ 0.,  1.,  1.,  0.]], dtype=float32), array([[ True,  True,  True,  True]], dtype=bool), 1.0]\n",
      "1.0\n"
     ]
    }
   ],
   "source": [
    "from __future__ import print_function\n",
    "# from builtins import range\n",
    "\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "\n",
    "xy = np.loadtxt('/data/hunkim/data_labs/lab_09/train.txt', unpack=True)\n",
    "# x_data = np.transpose(xy[0:-1])\n",
    "# y_data = np.transpose(xy[-1])\n",
    "x_data = xy[0:-1]\n",
    "y_data = xy[-1]\n",
    "\n",
    "print(x_data.shape, y_data.shape)\n",
    "\n",
    "X = tf.placeholder(tf.float32, name = 'X-input')\n",
    "Y = tf.placeholder(tf.float32, name = 'Y-input')\n",
    "\n",
    "# 2 - input, 2 - output\n",
    "W1 = tf.Variable(tf.random_uniform([5, 2], -1.0, 1.0), name = \"Weight1\")\n",
    "W2 = tf.Variable(tf.random_uniform([4, 5], -1.0, 1.0), name = \"Weight2\")\n",
    "# 2 - input, 1 - output\n",
    "W3 = tf.Variable(tf.random_uniform([1, 4], -1.0, 1.0), name = \"Weight3\")\n",
    "\n",
    "b1 = tf.Variable(tf.zeros([5, 1]), name=\"Bias1\")\n",
    "b2 = tf.Variable(tf.zeros([4, 1]), name=\"Bias2\")\n",
    "b3 = tf.Variable(tf.zeros([1]), name=\"Bias3\")\n",
    "\n",
    "L2 = tf.sigmoid(tf.matmul(W1, X) + b1)\n",
    "L3 = tf.sigmoid(tf.matmul(W2, L2) + b2)\n",
    "hypothesis = tf.sigmoid(tf.matmul(W3, L3) + b3)\n",
    "# hypothesis = tf.sigmoid(tf.matmul(W2, X) + b2)\n",
    "\n",
    "cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1 - hypothesis))\n",
    "\n",
    "a = tf.Variable(0.05)\n",
    "optimizer = tf.train.GradientDescentOptimizer(a)\n",
    "train = optimizer.minimize(cost)\n",
    "\n",
    "init = tf.initialize_all_variables()\n",
    "\n",
    "with tf.Session() as sess:\n",
    "    sess.run(init)\n",
    "    \n",
    "    merged = tf.merge_all_summaries()\n",
    "    writer = tf.train.SummaryWriter(\"./logs/xor_logs\", sess.graph_def)\n",
    "    \n",
    "    for step in range(20000):\n",
    "        sess.run(train, feed_dict={X:x_data, Y:y_data})\n",
    "        if step % 1000 == 0:\n",
    "#             summary = sess.run(merged, feed_dict={X:x_data, Y:y_data})\n",
    "#             writer.add_summary(summary, step)\n",
    "#             print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W1), sess.run(W2)\n",
    "            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))\n",
    "            \n",
    "    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)\n",
    "    accuracy = tf.reduce_mean(tf.cast(correct_prediction, \"float\"))\n",
    "    \n",
    "    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data}))\n",
    "\n",
    "#     print \"Accuracy: \" + accuracy.eval({X:x_data, Y:y_data})\n",
    "    print(accuracy.eval({X:x_data, Y:y_data}))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(4, 2) (4,)\n",
      "0 0.738092\n",
      "1000 0.693185\n",
      "2000 0.693182\n",
      "3000 0.693179\n",
      "4000 0.693177\n",
      "5000 0.693175\n",
      "6000 0.693173\n",
      "7000 0.693171\n",
      "8000 0.69317\n",
      "9000 0.693168\n",
      "10000 0.693167\n",
      "11000 0.693166\n",
      "12000 0.693164\n",
      "13000 0.693163\n",
      "14000 0.693162\n",
      "15000 0.693161\n",
      "16000 0.69316\n",
      "17000 0.693159\n",
      "18000 0.693159\n",
      "19000 0.693158\n",
      "[array([[ 0.49657467],\n",
      "       [ 0.50023305],\n",
      "       [ 0.5002594 ],\n",
      "       [ 0.50288737]], dtype=float32), array([[ 0.],\n",
      "       [ 1.],\n",
      "       [ 1.],\n",
      "       [ 1.]], dtype=float32), array([[ True, False, False,  True],\n",
      "       [False,  True,  True, False],\n",
      "       [False,  True,  True, False],\n",
      "       [False,  True,  True, False]], dtype=bool), 0.5]\n",
      "0.5\n"
     ]
    }
   ],
   "source": [
    "from __future__ import print_function\n",
    "# from builtins import range\n",
    "\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "\n",
    "xy = np.loadtxt('/data/hunkim/data_labs/lab_09/train.txt', unpack=True)\n",
    "x_data = np.transpose(xy[0:-1])\n",
    "# y_data = np.transpose(xy[-1])\n",
    "# x_data = xy[0:-1]\n",
    "y_data = xy[-1]\n",
    "\n",
    "print(x_data.shape, y_data.shape)\n",
    "\n",
    "X = tf.placeholder(tf.float32, name = 'X-input')\n",
    "Y = tf.placeholder(tf.float32, name = 'Y-input')\n",
    "\n",
    "# 2 - input, 2 - output\n",
    "W1 = tf.Variable(tf.random_uniform([2, 5], -1.0, 1.0), name = \"Weight1\")\n",
    "W2 = tf.Variable(tf.random_uniform([5, 4], -1.0, 1.0), name = \"Weight2\")\n",
    "# 2 - input, 1 - output\n",
    "W3 = tf.Variable(tf.random_uniform([4, 1], -1.0, 1.0), name = \"Weight3\")\n",
    "\n",
    "b1 = tf.Variable(tf.zeros([5]), name=\"Bias1\")\n",
    "b2 = tf.Variable(tf.zeros([4]), name=\"Bias2\")\n",
    "b3 = tf.Variable(tf.zeros([1]), name=\"Bias3\")\n",
    "\n",
    "L2 = tf.sigmoid(tf.matmul(X, W1) + b1)\n",
    "L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)\n",
    "hypothesis = tf.sigmoid(tf.matmul(L3, W3) + b3)\n",
    "# hypothesis = tf.sigmoid(tf.matmul(W2, X) + b2)\n",
    "\n",
    "cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1 - hypothesis))\n",
    "\n",
    "a = tf.Variable(0.05)\n",
    "optimizer = tf.train.GradientDescentOptimizer(a)\n",
    "train = optimizer.minimize(cost)\n",
    "\n",
    "init = tf.initialize_all_variables()\n",
    "\n",
    "with tf.Session() as sess:\n",
    "    sess.run(init)\n",
    "    \n",
    "    merged = tf.merge_all_summaries()\n",
    "    writer = tf.train.SummaryWriter(\"./logs/xor_logs\", sess.graph_def)\n",
    "    \n",
    "    for step in xrange(20000):\n",
    "        sess.run(train, feed_dict={X:x_data, Y:y_data})\n",
    "        if step % 1000 == 0:\n",
    "#             summary = sess.run(merged, feed_dict={X:x_data, Y:y_data})\n",
    "#             writer.add_summary(summary, step)\n",
    "#             print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W1), sess.run(W2)\n",
    "            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))\n",
    "            \n",
    "    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)\n",
    "    accuracy = tf.reduce_mean(tf.cast(correct_prediction, \"float\"))\n",
    "    \n",
    "    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data}))\n",
    "\n",
    "#     print \"Accuracy: \" + accuracy.eval({X:x_data, Y:y_data})\n",
    "    print(accuracy.eval({X:x_data, Y:y_data}))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(2, 4) (4,)\n",
      "0 0.73258\n",
      "1000 0.692172\n",
      "2000 0.689004\n",
      "3000 0.678713\n",
      "4000 0.631707\n",
      "5000 0.459829\n",
      "6000 0.241533\n",
      "7000 0.131391\n",
      "8000 0.0830035\n",
      "9000 0.0585105\n",
      "10000 0.0443376\n",
      "11000 0.0352943\n",
      "12000 0.0291008\n",
      "13000 0.0246296\n",
      "14000 0.021269\n",
      "15000 0.0186614\n",
      "16000 0.0165856\n",
      "17000 0.0148982\n",
      "18000 0.0135021\n",
      "19000 0.0123298\n",
      "[array([[ 0.00694776,  0.98884958,  0.98782605,  0.01479065]], dtype=float32), array([[ 0.,  1.,  1.,  0.]], dtype=float32), array([[ True,  True,  True,  True]], dtype=bool), 1.0]\n",
      "1.0\n"
     ]
    }
   ],
   "source": [
    "from __future__ import print_function\n",
    "\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "\n",
    "xy = np.loadtxt('/data/hunkim/data_labs/lab_09/train.txt', unpack=True)\n",
    "# x_data = np.transpose(xy[0:-1])\n",
    "# y_data = np.transpose(xy[-1])\n",
    "x_data = xy[0:-1]\n",
    "y_data = xy[-1]\n",
    "\n",
    "print(x_data.shape, y_data.shape)\n",
    "\n",
    "X = tf.placeholder(tf.float32, name = 'X-input')\n",
    "Y = tf.placeholder(tf.float32, name = 'Y-input')\n",
    "\n",
    "# 2 - input, 2 - output\n",
    "W1 = tf.Variable(tf.random_uniform([10, 2], -1.0, 1.0), name = \"Weight1\")\n",
    "# 2 - input, 1 - output\n",
    "W2 = tf.Variable(tf.random_uniform([1, 10], -1.0, 1.0), name = \"Weight2\")\n",
    "\n",
    "b1 = tf.Variable(tf.zeros([10, 1]), name=\"Bias1\")\n",
    "b2 = tf.Variable(tf.zeros([1]), name=\"Bias2\")\n",
    "\n",
    "L2 = tf.sigmoid(tf.matmul(W1, X) + b1)\n",
    "hypothesis = tf.sigmoid(tf.matmul(W2, L2) + b2)\n",
    "# hypothesis = tf.sigmoid(tf.matmul(W2, X) + b2)\n",
    "\n",
    "cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1 - hypothesis))\n",
    "\n",
    "a = tf.Variable(0.05)\n",
    "optimizer = tf.train.GradientDescentOptimizer(a)\n",
    "train = optimizer.minimize(cost)\n",
    "\n",
    "init = tf.initialize_all_variables()\n",
    "\n",
    "with tf.Session() as sess:\n",
    "    sess.run(init)\n",
    "    \n",
    "    merged = tf.merge_all_summaries()\n",
    "    writer = tf.train.SummaryWriter(\"./logs/xor_logs\", sess.graph_def)\n",
    "    \n",
    "    for step in xrange(20000):\n",
    "        sess.run(train, feed_dict={X:x_data, Y:y_data})\n",
    "        if step % 1000 == 0:\n",
    "#             summary = sess.run(merged, feed_dict={X:x_data, Y:y_data})\n",
    "#             writer.add_summary(summary, step)\n",
    "#             print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W1), sess.run(W2)\n",
    "            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))\n",
    "            \n",
    "    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)\n",
    "    accuracy = tf.reduce_mean(tf.cast(correct_prediction, \"float\"))\n",
    "    \n",
    "    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data}))\n",
    "\n",
    "#     print \"Accuracy: \" + accuracy.eval({X:x_data, Y:y_data})\n",
    "    print(accuracy.eval({X:x_data, Y:y_data}))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## TensorBoard"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": false,
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(2, 4) (4,)\n",
      "0 0.693334\n",
      "1000 0.693064\n",
      "2000 0.692834\n",
      "3000 0.692575\n",
      "4000 0.692247\n",
      "5000 0.691783\n",
      "6000 0.691059\n",
      "7000 0.689812\n",
      "8000 0.687384\n",
      "9000 0.68184\n",
      "10000 0.666044\n",
      "11000 0.60622\n",
      "12000 0.34914\n",
      "13000 0.0943746\n",
      "14000 0.0394723\n",
      "15000 0.0229777\n",
      "16000 0.0157132\n",
      "17000 0.0117603\n",
      "18000 0.00931602\n",
      "19000 0.00767113\n",
      "[array([[ 0.00691466,  0.99300718,  0.9936381 ,  0.00563317]], dtype=float32), array([[ 0.,  1.,  1.,  0.]], dtype=float32), array([[ True,  True,  True,  True]], dtype=bool), 1.0]\n",
      "1.0\n"
     ]
    }
   ],
   "source": [
    "from __future__ import print_function\n",
    "# from builtins import range\n",
    "\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "\n",
    "xy = np.loadtxt('/data/hunkim/data_labs/lab_09/train.txt', unpack=True)\n",
    "# x_data = np.transpose(xy[0:-1])\n",
    "# y_data = np.transpose(xy[-1])\n",
    "x_data = xy[0:-1]\n",
    "y_data = xy[-1]\n",
    "\n",
    "print(x_data.shape, y_data.shape)\n",
    "\n",
    "X = tf.placeholder(tf.float32, name = 'X-input')\n",
    "Y = tf.placeholder(tf.float32, name = 'Y-input')\n",
    "\n",
    "# 2 - input, 2 - output\n",
    "W1 = tf.Variable(tf.random_uniform([5, 2], -1.0, 1.0), name = \"Weight1\")\n",
    "W2 = tf.Variable(tf.random_uniform([4, 5], -1.0, 1.0), name = \"Weight2\")\n",
    "# 2 - input, 1 - output\n",
    "W3 = tf.Variable(tf.random_uniform([1, 4], -1.0, 1.0), name = \"Weight3\")\n",
    "\n",
    "b1 = tf.Variable(tf.zeros([5, 1]), name=\"Bias1\")\n",
    "b2 = tf.Variable(tf.zeros([4, 1]), name=\"Bias2\")\n",
    "b3 = tf.Variable(tf.zeros([1]), name=\"Bias3\")\n",
    "\n",
    "# L2 = tf.sigmoid(tf.matmul(W1, X) + b1)\n",
    "# L3 = tf.sigmoid(tf.matmul(W2, L2) + b2)\n",
    "# hypothesis = tf.sigmoid(tf.matmul(W3, L3) + b3)\n",
    "# cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1 - hypothesis))\n",
    "# a = tf.Variable(0.05)\n",
    "# optimizer = tf.train.GradientDescentOptimizer(a)\n",
    "# train = optimizer.minimize(cost)\n",
    "\n",
    "\n",
    "tf.histogram_summary(\"weights1\", W1)\n",
    "tf.histogram_summary(\"weights2\", W2)\n",
    "tf.histogram_summary(\"weights3\", W3)\n",
    "\n",
    "tf.histogram_summary(\"biases1\", b1)\n",
    "tf.histogram_summary(\"biases2\", b2)\n",
    "tf.histogram_summary(\"biases3\", b3)\n",
    "\n",
    "tf.histogram_summary(\"y\", Y)\n",
    "\n",
    "\n",
    "with tf.name_scope(\"layer1\") as scope:\n",
    "    L2 = tf.sigmoid(tf.matmul(W1, X) + b1)\n",
    "\n",
    "with tf.name_scope(\"layer2\") as scope:\n",
    "    L3 = tf.sigmoid(tf.matmul(W2, L2) + b2)\n",
    "\n",
    "with tf.name_scope(\"layer3\") as scope:\n",
    "    hypothesis = tf.sigmoid(tf.matmul(W3, L3) + b3)\n",
    "# hypothesis = tf.sigmoid(tf.matmul(W2, X) + b2)\n",
    "\n",
    "with tf.name_scope(\"cost\") as scope:\n",
    "    cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1 - hypothesis))\n",
    "    cost_summ = tf.scalar_summary(\"cost\", cost)\n",
    "\n",
    "with tf.name_scope(\"train\") as scope:\n",
    "    a = tf.Variable(0.05)\n",
    "    optimizer = tf.train.GradientDescentOptimizer(a)\n",
    "    train = optimizer.minimize(cost)\n",
    "\n",
    "init = tf.initialize_all_variables()\n",
    "\n",
    "with tf.Session() as sess:\n",
    "    sess.run(init)\n",
    "    \n",
    "    merged = tf.merge_all_summaries()\n",
    "    writer = tf.train.SummaryWriter(\"./logs/xor_logs\", sess.graph_def)\n",
    "    \n",
    "    for step in range(20000):\n",
    "        sess.run(train, feed_dict={X:x_data, Y:y_data})\n",
    "        if step % 100 == 0:\n",
    "            summary = sess.run(merged, feed_dict={X:x_data, Y:y_data})\n",
    "            writer.add_summary(summary, step)\n",
    "#             print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W1), sess.run(W2)\n",
    "            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))\n",
    "            \n",
    "    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)\n",
    "    accuracy = tf.reduce_mean(tf.cast(correct_prediction, \"float\"))\n",
    "    \n",
    "    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data}))\n",
    "\n",
    "#     print \"Accuracy: \" + accuracy.eval({X:x_data, Y:y_data})\n",
    "    print(accuracy.eval({X:x_data, Y:y_data}))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "tensorboard --logdir=./logs/xor_logs<br>\n",
    "http://0.0.0.0:6006\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
