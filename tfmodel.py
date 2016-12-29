#!/usr/bin/evn python
import tensorflow as tf


class TFModel(object):
    def __init__(self, session=None, nid="tfm", input_data = None, verbose=True, reuse=False):
        if not session:
            session = tf.Session()
        self._session = session
        self.nid = nid    # network id string
        self._verbose = verbose
        self._reuse = reuse
        self._model = None
        self._is_initialized = False  # are the variables already initialized?
        self._input_data = input_data

    @property
    def session(self):
        return self._session

    @property
    def model(self):
        return self._model

    @property
    def input_data(self):
        return self._input_data

    def _build_model(self):
        pass

    def _initialize_variables(self):
        # initialize the variables if not done already
        if not self._is_initialized:
            # init = tf.global_variables_initializer()
            init = tf.initialize_all_variables()
            self._session.run(init)
            self._is_initialized = True
