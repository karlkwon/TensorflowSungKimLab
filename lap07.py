{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "https://github.com/aymericdamien/TensorFlow-Examples"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "total 12544\r\n",
      "drwxrwxr-x 6 1000  1000    4096 Dec  7 08:10 .\r\n",
      "drwxrwxr-x 5 1000  1000    4096 Apr 21 15:11 ..\r\n",
      "-rw-r--r-- 1 1000  1000       0 Nov 25 15:06 __init__.py\r\n",
      "drwxr-xr-x 3 1000  1000    4096 Nov 25 15:06 beginners\r\n",
      "drwxr-xr-x 2 1000  1000    4096 Nov 25 15:06 download\r\n",
      "-rw-r--r-- 1 1000  1000    8827 Nov 25 15:06 fully_connected_feed.py\r\n",
      "-rw-r--r-- 1 1000  1000    6582 Nov 25 15:06 input_data.py\r\n",
      "-rw-r--r-- 1 1000  1000    6113 Nov 25 15:06 mnist.py\r\n",
      "-rw-r--r-- 1 1000  1000    1804 Nov 25 15:06 mnist_softmax.py\r\n",
      "drwxr-xr-x 2 1000  1000    4096 Nov 25 15:06 pros\r\n",
      "-rw-r--r-- 1 1000 users 1648877 Dec  7 08:10 t10k-images-idx3-ubyte.gz\r\n",
      "-rw-r--r-- 1 1000 users    4542 Dec  7 08:10 t10k-labels-idx1-ubyte.gz\r\n",
      "-rw-rw-r-- 1 1000  1000 1179565 Nov 25 15:06 tensorflow-master-tensorflow-g3doc-tutorials-mnist.tar.gz\r\n",
      "drwxr-xr-x 2 1000  1000    4096 Nov 25 15:06 tf\r\n",
      "-rw-r--r-- 1 1000 users 9912422 Dec  7 08:10 train-images-idx3-ubyte.gz\r\n",
      "-rw-r--r-- 1 1000 users   28881 Dec  7 08:10 train-labels-idx1-ubyte.gz\r\n"
     ]
    }
   ],
   "source": [
    "!ls -al /data/mnist"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "!cp /data/TensorFlow-Examples/input_data.py ./."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting /data/mnist/train-images-idx3-ubyte.gz\n",
      "Extracting /data/mnist/train-labels-idx1-ubyte.gz\n",
      "Extracting /data/mnist/t10k-images-idx3-ubyte.gz\n",
      "Extracting /data/mnist/t10k-labels-idx1-ubyte.gz\n"
     ]
    }
   ],
   "source": [
    "import input_data\n",
    "mnist = input_data.read_data_sets(\"/data/mnist\", one_hot=True)"
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
     "data": {
      "text/plain": [
       "input_data.DataSets"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(mnist)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "learning_rate = 0.001\n",
    "training_iters = 1000\n",
    "batch_size = 128\n",
    "display_step = 10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "training_epochs = 100\n",
    "\n",
    "n_input = 784\n",
    "n_classes = 10\n",
    "# dropout = 0.75"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "x = tf.placeholder(tf.float32, [None, n_input])\n",
    "y = tf.placeholder(tf.float32, [None, n_classes])\n",
    "keep_prob = tf.placeholder(tf.float32)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "W = tf.Variable(tf.zeros([784, 10]))\n",
    "b = tf.Variable(tf.zeros([10]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "activation = tf.nn.softmax(tf.matmul(x, W) + b)\n",
    "\n",
    "cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1))\n",
    "optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)\n",
    "\n",
    "init = tf.initialize_all_variables()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch:  0001 cost=  2.095915547\n",
      "Epoch:  0011 cost=  0.838093956\n",
      "Epoch:  0021 cost=  0.641112747\n",
      "Epoch:  0031 cost=  0.558944515\n",
      "Epoch:  0041 cost=  0.511981974\n",
      "Epoch:  0051 cost=  0.481237279\n",
      "Epoch:  0061 cost=  0.458905554\n",
      "Epoch:  0071 cost=  0.441659588\n",
      "Epoch:  0081 cost=  0.428204215\n",
      "Epoch:  0091 cost=  0.416850964\n",
      "Optimization Finished !\n"
     ]
    }
   ],
   "source": [
    "sess = tf.Session()\n",
    "\n",
    "# with tf.Session() as sess:\n",
    "sess.run(init)\n",
    "\n",
    "costs = []\n",
    "for epoch in range(training_epochs):\n",
    "    avg_cost = 0.\n",
    "    total_batch = int(mnist.train.num_examples/batch_size)\n",
    "\n",
    "    for i in range(total_batch):\n",
    "        batch_xs, batch_ys = mnist.train.next_batch(batch_size)\n",
    "        sess.run(optimizer, feed_dict={x: batch_xs, y:batch_ys})\n",
    "        avg_cost += sess.run(cost, feed_dict={x:batch_xs, y:batch_ys})/total_batch\n",
    "        \n",
    "    costs.append(avg_cost)\n",
    "    if epoch % display_step == 0:\n",
    "        print \"Epoch: \", '%04d' % (epoch+1), \"cost= \", \"{:.9f}\".format(avg_cost)\n",
    "\n",
    "print \"Optimization Finished !\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[2.0959155470619111, 1.7638561864555029, 1.523129973378214, 1.3466443959927503, 1.215049672515798, 1.1139825176803659, 1.0349144723031898, 0.97112699210782671, 0.91891117960145141, 0.87519764469497996, 0.83809395561685052, 0.80631904779892138, 0.77864485173236453, 0.75434603015859614, 0.73269237102050622, 0.71352357214147388, 0.69599540889402078, 0.68015983193626517, 0.66623641394235056, 0.65329693608628514, 0.64111274717015232, 0.63016231847809767, 0.62013606945951494, 0.6106112398606639, 0.60160051706509698, 0.59327083034115291, 0.58573772759982201, 0.57825232823411921, 0.57161997595589176, 0.56511457677765642, 0.55894451548447355, 0.5532686631162681, 0.54780549998883554, 0.54242691383773112, 0.53760261246652319, 0.53286662753367497, 0.52856678416678948, 0.52407925008060241, 0.52000446426562819, 0.51605987923962271, 0.51198197404543533, 0.50854720116217422, 0.50508477061222801, 0.50174851079920757, 0.49830575867410587, 0.4951098680774087, 0.4924959559818527, 0.48938225935666996, 0.48662167426311626, 0.48381224059280498, 0.48123727916003922, 0.47871084598116659, 0.47621104003110371, 0.47376242919123801, 0.47156121486272579, 0.46910152770144714, 0.46697661252844302, 0.46494526379591899, 0.46286879337473064, 0.46088050920646534, 0.45890555417898915, 0.45709248710345535, 0.45489531630402624, 0.45327217956800797, 0.45165296569292951, 0.44994582308755876, 0.44826717004353644, 0.44656917890468639, 0.44490303998782627, 0.44322481035908323, 0.44165958821912465, 0.44034877086019203, 0.43896623438610644, 0.43741154934698606, 0.43594202920273512, 0.4345271584593059, 0.43322441322264449, 0.43191814603227568, 0.43062894765313697, 0.42941439776987483, 0.42820421451733082, 0.42708357015411863, 0.42563662252503986, 0.42457597475229697, 0.42340799707632798, 0.42231705673646786, 0.42135809910741967, 0.42000370773124268, 0.41899039347966549, 0.41821768831261924, 0.41685096388096554, 0.41611872908674441, 0.41491802932915955, 0.41400883358953128, 0.41286453437971887, 0.41199272488936428, 0.41126721312393977, 0.41041028433607446, 0.40938088006072937, 0.40861405004988144]\n"
     ]
    }
   ],
   "source": [
    "print(costs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\ngg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fc8f02fb250>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(costs, 'ro')\n",
    "plt.ylabel('Cost')\n",
    "plt.xlabel('t')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.])"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mnist.test.labels[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " Label:  [9]\n",
      "Prediction:  [9]\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAP4AAAD8CAYAAABXXhlaAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAADnpJREFUeJzt3X+MF/Wdx/HXWzYXU9CKzQGJe7VezOGlCcHikRAuOsRe\nSy6NmKocJ39o72IaUu9KarRooruc90erhsT7g0QFKpiSgk0o9FSURr9pxPTEO7lDi0vNBYsg63rh\nDAtEKL7vjx32lmW/n/nuzvfHLO/nI9kwO+/Z77wZeO3MfD/fmTF3F4BYLul0AwDaj+ADARF8ICCC\nDwRE8IGACD4QUKngm9liM3vPzA6Y2Q+b1RSA1rKJjuOb2SWSDki6WdIRSXskLXP390YtxwcFgA5x\ndxtrfpk9/nxJv3P3D9z9jKSfSVpSZ+XDXz09Ped9X7Uv+rt4+6tyb63oL6VM8K+SdGjE9x/m8wBU\nHG/uAQF1lfjZw5K+POL77nzeBXp7e4enr7jiihKrbL0syzrdQhL9TVyVe5PK91er1VSr1Rpatsyb\ne1Mk9Wnozb2PJL0p6W/dff+o5Xyi6wAwcWYmr/Pm3oT3+O5+1szulfSKhk4Z1o8OPYBqmvAev+EV\nsMcHOiK1x+fNPSAggg8ERPCBgAg+EBDBBwIi+EBABB8IiOADARF8ICCCDwRE8IGACD4QEMEHAiL4\nQEAEHwiI4AMBEXwgIIIPBETwgYAIPhAQwQcCIvhAQAQfCIjgAwERfCAggg8ERPCBgAg+EBDBBwIi\n+EBABB8IqKvMD5vZQUmfSvpc0hl3n9+MpgC0Vqngayjwmbsfa0YzANqj7KG+NeE1ALRZ2dC6pF1m\ntsfM7mlGQwBar+yh/kJ3/8jM/lhDvwD2u/vroxfq7e0dns6yTFmWlVwtgNFqtZpqtVpDy5q7N2Wl\nZtYj6bi7rxk135u1DgCNMzO5u41Vm/Chvpl9wcym5dNTJX1D0jsTfT0A7VPmUH+mpG1m5vnr/NTd\nX2lOWwBaqWmH+nVXwKE+0BEtOdQHMHkRfCAggg8ERPCBgAg+EBDBBwIi+EBAZT+rj0nu9OnTyXpf\nX1+y/uKLLybrDz744Lh7GqnoMyBmYw5TD7vuuuuS9bVr1ybrN910U6n1VxV7fCAggg8ERPCBgAg+\nEBDBBwIi+EBABB8IiOvxJ7mTJ08m688++2yyvmrVqmT9xIkTyXrZcfaq27ZtW7J+yy23tKmT8eN6\nfADnIfhAQAQfCIjgAwERfCAggg8ERPCBgLgev8OOHz+erG/atClZf/TRR5P1gYGBcfc0HjNmzEjW\nZ8+enaw/8sgjzWznAo8//niyvmvXrmR99+7dyXqVx/FT2OMDARF8ICCCDwRE8IGACD4QEMEHAiL4\nQECF4/hmtl7StyT1u/ucfN50SVskXS3poKSl7v5pC/vsmLNnzybrRfel/+yzz5L1OXPmJOuHDx9O\n1suaP39+sr548eJkfcWKFcl60Th/kTNnziTr/f39yfrmzZuT9eXLl4+7p4tBI3v8n0j65qh5qyT9\nyt1nS3pVUrmnJgBoq8Lgu/vrko6Nmr1E0sZ8eqOkW5vcF4AWmug5/gx375ckdz8qqdzxHIC2atZn\n9ZM3Xuvt7R2ezrJMWZY1abUAzqnVaqrVag0tO9Hg95vZTHfvN7NZkj5OLTwy+ABaY/ROdfXq1XWX\nbfRQ3/Kvc3ZIujufvkvS9vE0CKCzCoNvZpslvSHpz8zs92b2HUk/kvRXZtYn6eb8ewCTROGhvrvf\nWaf09Sb3UklF940vGmfesGFDsl52nH7mzJnJ+pYtW5L1BQsWJOtdXa29ZUPR9nv44YeT9aLr7Xt6\nepL1559/Plkvul/CZMUn94CACD4QEMEHAiL4QEAEHwiI4AMBEXwgIO6rX6BoHPvyyy9P1t9///1m\ntnOBF154IVm//vrrW7r+sp5++ulkvWicvqxp06aVqk9W7PGBgAg+EBDBBwIi+EBABB8IiOADARF8\nICArut689ArMvNXr6KR9+/Yl64sWLUrWjx0bfQPj8Sm6L/9ll12WrBc9n37q1KnJ+s6dO5P11157\nLVk/cOBAsj4wMJCsF+nr60vWr7322lKvX2VmJne3sWrs8YGACD4QEMEHAiL4QEAEHwiI4AMBEXwg\nIMbxSzpx4kSy/tRTTyXr999/fzPbabuif1uzMYeR24ZxfMbxAeQIPhAQwQcCIvhAQAQfCIjgAwER\nfCCgwnF8M1sv6VuS+t19Tj6vR9I9kj7OF3vI3ce8MPtiH8cvcvbs2WT91KlTyfrtt9+erO/atWvc\nPTVTq8fxZ82alaw/9thjyfqyZcuS9SlTpoy7p8mi7Dj+TyR9c4z5a9z9a/lX+m4MACqlMPju/rqk\nsW4T09mPZAGYsDLn+Pea2V4zW2dmX2xaRwBabqLPzlsr6Z/c3c3snyWtkfT39Rbu7e0dns6yTFmW\nTXC1AOqp1Wqq1WoNLTuh4Lv7yDsgPiPpl6nlRwYfQGuM3qmuXr267rKNHuqbRpzTm9nIt1q/Lemd\ncXUIoKMK9/hmtllSJulLZvZ7ST2SFpnZXEmfSzoo6bst7BFAk3E9fsWdOXMmWd++fXuyfuTIkVLr\nv/HGG5P106dPJ+sLFiwotf6i+/739PSUev2LGdfjAzgPwQcCIvhAQAQfCIjgAwERfCAggg8ExDg+\nkoruS79y5cpk/eWXX07Wu7u7k/W33norWZ8xY0ayHhnj+ADOQ/CBgAg+EBDBBwIi+EBABB8IiOAD\nATGOj6TbbrstWd+2bVuyXnTf+qJ7xC1cuDBZR32M4wM4D8EHAiL4QEAEHwiI4AMBEXwgIIIPBMQ4\nfnBvvPFGsr5o0aJkvei+//PmzUvW9+zZk6xj4hjHB3Aegg8ERPCBgAg+EBDBBwIi+EBABB8IqHAc\n38y6JW2SNFPS55Kecfd/MbPpkrZIulrSQUlL3f3TMX6ecfwOOnXqVLJ+zTXXJOsDAwPJ+qWXXpqs\nHzp0KFm/8sork3VMXNlx/D9I+oG7f1XSAknfM7PrJK2S9Ct3ny3pVUkPNqthAK1VGHx3P+rue/Pp\nQUn7JXVLWiJpY77YRkm3tqpJAM01rnN8M/uKpLmSfiNpprv3S0O/HCTxLCNgkuhqdEEzmybp55K+\n7+6DZjb6xL3uiXxvb+/wdJZlyrJsfF0CKFSr1QrvYXhOQxfpmFmXpH+V9JK7P5nP2y8pc/d+M5sl\n6TV3//MxfpY39zqIN/fiasZFOhsk/fZc6HM7JN2dT98lafuEOwTQVoWH+ma2UNJySfvM7G0NHdI/\nJOnHkraa2d9J+kDS0lY2CqB5CoPv7rsl1bs5+teb2w7G6+TJk8n68uXLk/WiQ/kiTzzxRLLOoXw1\n8ck9ICCCDwRE8IGACD4QEMEHAiL4QEAEHwio4c/qo5q2bt2arO/YsaPU669cuTJZX7FiRanXR2ew\nxwcCIvhAQAQfCIjgAwERfCAggg8ERPCBgBq69VapFXDrrVJ27tyZrN9xxx3JetH1+kU++eSTZH36\n9OmlXh+t04xbbwG4iBB8ICCCDwRE8IGACD4QEMEHAiL4QEBcj99hg4ODyfoDDzyQrJcdpy9y+PDh\nZJ1x/MmJPT4QEMEHAiL4QEAEHwiI4AMBEXwgoMLgm1m3mb1qZu+a2T4z+4d8fo+ZfWhm/5F/LW59\nuwCaoZFx/D9I+oG77zWzaZL+3cx25bU17r6mde1d/O67775k/d133y31+jfccEOy/tJLLyXrjNNf\nnAqD7+5HJR3NpwfNbL+kq/LymBf5A6i2cZ3jm9lXJM2V9G/5rHvNbK+ZrTOzLza5NwAt0nDw88P8\nn0v6vrsPSlor6U/dfa6Gjgg45AcmiYY+q29mXRoK/XPuvl2S3H1gxCLPSPplvZ/v7e0dns6yTFmW\nTaBVACm1Wk21Wq2hZRu9SGeDpN+6+5PnZpjZrPz8X5K+Lemdej88MvgAWmP0TnX16tV1ly0Mvpkt\nlLRc0j4ze1uSS3pI0p1mNlfS55IOSvpumaYBtE8j7+rvljRljFL6vs8AKovr8Tts3rx5yfq6deuS\n9eeeey5ZX7p0abLe1cV/gYj4yC4QEMEHAiL4QEAEHwiI4AMBEXwgIIIPBGStfna9mXmr1wHgQmYm\ndx/z0nn2+EBABB8IiOADAbU9+I1eL9wp9FdOlfurcm9Se/sj+KPQXzlV7q/KvUkXefABdB7BBwJq\nyzh+S1cAoK564/gtDz6A6uFQHwiI4AMBtS34ZrbYzN4zswNm9sN2rbdRZnbQzP7TzN42szcr0M96\nM+s3s/8aMW+6mb1iZn1m9nInn15Up7/KPEh1jIe9/mM+vxLbsNMPo23LOb6ZXSLpgKSbJR2RtEfS\nMnd/r+Urb5CZ/bekee5+rNO9SJKZ/aWkQUmb3H1OPu/Hkv7H3R/Lf3lOd/dVFeqvR9LxKjxI1cxm\nSZo18mGvkpZI+o4qsA0T/f2N2rAN27XHny/pd+7+gbufkfQzDf0lq8RUoVMfd39d0uhfQkskbcyn\nN0q6ta1NjVCnP6kiD1J196PuvjefHpS0X1K3KrIN6/TXtofRtus/+lWSDo34/kP9/1+yKlzSLjPb\nY2b3dLqZOma4e780/BTjGR3uZyyVe5DqiIe9/kbSzKptw048jLYye7gKWOjuX5P015K+lx/KVl3V\nxmIr9yDVMR72OnqbdXQbduphtO0K/mFJXx7xfXc+rzLc/aP8zwFJ2zR0elI1/WY2Uxo+R/y4w/2c\nx90HRtx15RlJf9HJfsZ62KsqtA3rPYy2HduwXcHfI+laM7vazP5I0jJJO9q07kJm9oX8N6/MbKqk\nbyjxENA2Mp1/vrdD0t359F2Sto/+gTY7r788SOckH6TaJhc87FXV2oZjPox2RL1l27Btn9zLhyWe\n1NAvm/Xu/qO2rLgBZnaNhvbyrqHHiv200/2Z2WZJmaQvSeqX1CPpF5Kel/Qnkj6QtNTd/7dC/S3S\n0Lnq8INUz51Pd6C/hZJ+LWmfhv5dzz3s9U1JW9XhbZjo7061YRvykV0gIN7cAwIi+EBABB8IiOAD\nARF8ICCCDwRE8IGACD4Q0P8Bn0kkcLpZNMEAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fc8d56eb890>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from random import randint\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "r = randint(0, mnist.test.num_examples - 1)\n",
    "# maybe labels is a 1-hot vector.\n",
    "# So, argmax retun a label index.\n",
    "print \"Label: \", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1))\n",
    "print \"Prediction: \", sess.run(tf.argmax(activation, 1), {x:mnist.test.images[r:r+1]})\n",
    "\n",
    "plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy:  0.8993\n"
     ]
    }
   ],
   "source": [
    "correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))\n",
    "accuracy = tf.reduce_mean(tf.cast(correct_prediction, \"float\"))\n",
    "print \"Accuracy: \", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}, session=sess)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "sess.close()"
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
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from __future__ import print_function\n",
    "from builtins import range\n",
    "import input_data\n",
    "import tensorflow as tf\n",
    "\n",
    "mnist = input_data.read_data_sets(\"/data/mnist\", one_hot=True)\n",
    "\n",
    "learning_rate = 0.001\n",
    "training_iters = 1000\n",
    "batch_size = 128\n",
    "display_step = 10\n",
    "\n",
    "training_epochs = 100\n",
    "\n",
    "n_input = 784\n",
    "n_classes = 10\n",
    "# dropout = 0.75\n",
    "\n",
    "x = tf.placeholder(tf.float32, [None, n_input])\n",
    "y = tf.placeholder(tf.float32, [None, n_classes])\n",
    "keep_prob = tf.placeholder(tf.float32)\n",
    "\n",
    "W = tf.Variable(tf.zeros([784, 10]))\n",
    "b = tf.Variable(tf.zeros([10]))\n",
    "\n",
    "activation = tf.nn.softmax(tf.matmul(x, W) + b)\n",
    "\n",
    "cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1))\n",
    "optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)\n",
    "\n",
    "init = tf.initialize_all_variables()\n",
    "\n",
    "with tf.Session() as sess:\n",
    "    sess.run(init)\n",
    "\n",
    "    costs = []\n",
    "    for epoch in range(training_epochs):\n",
    "        avg_cost = 0.\n",
    "        total_batch = int(mnist.train.num_examples/batch_size)\n",
    "\n",
    "        for i in range(total_batch):\n",
    "            batch_xs, batch_ys = mnist.train.next_batch(batch_size)\n",
    "            sess.run(optimizer, feed_dict={x: batch_xs, y:batch_ys})\n",
    "            avg_cost += sess.run(cost, feed_dict={x:batch_xs, y:batch_ys})/total_batch\n",
    "\n",
    "        costs.append(avg_cost)\n",
    "        if epoch % display_step == 0:\n",
    "            print(\"Epoch: \", '%04d' % (epoch+1), \"cost= \", \"{:.9f}\".format(avg_cost))\n",
    "\n",
    "    print(\"Optimization Finished !\")\n",
    "\n",
    "    # Test model\n",
    "    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))\n",
    "    \n",
    "    #Calculate accuracy\n",
    "    accuracy = tf.reduce_mean(tf.cast(correct_prediction, \"float\"))\n",
    "    print(\"Accuracy:\", accuracy.eval({x: mnist.test.images, y: mnist.test.label}))\n",
    "    "
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
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "ename": "IOError",
     "evalue": "[Errno 2] No such file or directory: '/data/hunkim/data_labs/lab_07/train.txt'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mIOError\u001b[0m                                   Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-4aa0331f5d82>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m()\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mtensorflow\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mtf\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 4\u001b[1;33m \u001b[0mxy\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mloadtxt\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'/data/hunkim/data_labs/lab_07/train.txt'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0munpack\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mTrue\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'float32'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      5\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      6\u001b[0m \u001b[0mtraining_epochs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m25\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m/usr/lib/python2.7/dist-packages/numpy/lib/npyio.pyc\u001b[0m in \u001b[0;36mloadtxt\u001b[1;34m(fname, dtype, comments, delimiter, converters, skiprows, usecols, unpack, ndmin)\u001b[0m\n\u001b[0;32m    732\u001b[0m                 \u001b[0mfh\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0miter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mbz2\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mBZ2File\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfname\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    733\u001b[0m             \u001b[1;32melif\u001b[0m \u001b[0msys\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mversion_info\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;36m2\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 734\u001b[1;33m                 \u001b[0mfh\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0miter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mopen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfname\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'U'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    735\u001b[0m             \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    736\u001b[0m                 \u001b[0mfh\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0miter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mopen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfname\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mIOError\u001b[0m: [Errno 2] No such file or directory: '/data/hunkim/data_labs/lab_07/train.txt'"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "\n",
    "xy = np.loadtxt('/data/hunkim/data_labs/lab_07/train.txt', unpack=True, dtype='float32')\n",
    "\n",
    "training_epochs = 25\n",
    "batch_size = 100\n",
    "display_step = 1\n",
    "\n",
    "X = tf.placeholder(\"float\", [None, 784])\n",
    "Y = tf.placeholder(\"float\", [None, 10])\n",
    "\n",
    "W = tf.Variable(tf.zeros([784, 10]))\n",
    "b = tf.Variable(tf.zeros([10]))\n",
    "\n",
    "activation = tf.nn.softmax(tf.matmul(x, W) + b)\n",
    "\n",
    "cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indeces=1))\n",
    "optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)\n",
    "\n",
    "init = tf.initialize_all_variables()\n",
    "\n",
    "with tf.Session() as sess:\n",
    "    sess.run(init)\n",
    "    \n",
    "    for epoch in range(training_epochs):\n",
    "        avg_cost = 0.\n",
    "        total_batch = int(mnist.train.num_examples/batch_size)\n",
    "        \n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'06/02/16 14:17:49'"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from time import localtime, strftime\n",
    "\n",
    "strftime(\"%x %X\", localtime())"
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
