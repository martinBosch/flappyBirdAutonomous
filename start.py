#!/usr/bin/env python

from agent import Agent
import tensorflow as tf
if __name__ == "__main__":

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    Agent().run()
