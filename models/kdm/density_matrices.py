import numpy as np
import tensorflow as tf


def dm2comp(dm):
    """
    Extract vectors and weights from a factorized density matrix representation
    
    Args:
        dm: tensor of shape (bs, n, d + 1)
    
    Returns:
        w: tensor of shape (bs, n)
        v: tensor of shape (bs, n, d)
    """
    return dm[:, :, 0], dm[:, :, 1:]


def comp2dm(w, v):
    """
    Construct a factorized density matrix from vectors and weights
    
    Args:
        w: tensor of shape (bs, n)
        v: tensor of shape (bs, n, d)
    
    Returns:
        dm: tensor of shape (bs, n, d + 1)
    """
    return tf.concat((w[:, :, tf.newaxis], v), axis=2)


def samples2dm(samples):
    """
    Construct a factorized density matrix from a batch of samples
    
    Args:
        samples: tensor of shape (bs, n, d)
    
    Returns:
        dm: tensor of shape (bs, n, d + 1)
    """
    w = tf.reduce_any(samples, axis=-1)
    w = w / tf.reduce_sum(w, axis=-1, keepdims=True)
    return comp2dm(w, samples)


def pure2dm(psi):
    """
    Construct a factorized density matrix to represent a pure state
    
    Args:
        psi: tensor of shape (bs, d)
    
    Returns:
        dm: tensor of shape (bs, 1, d + 1)
    """
    ones = tf.ones_like(psi[:, 0:1])
    dm = tf.concat((ones[:, tf.newaxis, :],
                    psi[:, tf.newaxis, :]),
                   axis=2)
    return dm


def dm2discrete(dm):
    """
    Creates a discrete distribution from the components of a density matrix
    
    Args:
        dm: tensor of shape (bs, n, d + 1)
    
    Returns:
        prob: vector of probabilities (bs, d)
    """
    w, v = dm2comp(dm)
    w = w / tf.reduce_sum(w, axis=-1, keepdims=True)
    norms_v = tf.expand_dims(tf.linalg.norm(v, axis=-1), axis=-1)
    v = v / norms_v
    probs = tf.einsum('...j,...ji->...i', w, v ** 2, optimize="optimal")
    return probs