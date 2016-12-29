#!/usr/bin/evn python
import tensorflow as tf


class TFModel(object):
    def __init__(self, session=None, nid="tfm", verbose=True, reuse=False):
        if not session:
            session = tf.Session()
        self._session = session
        self.nid = nid    # network id string
        self._verbose = verbose
        self._reuse = reuse
        self._model = None
        self._is_initialized = False  # are the variables already initialized?
        self._input_data = None  # placeholder for input data - None in tfmodel

    @property
    def session(self):
        return self._session

    @property
    def model(self):
        return self._model

    def _build_model(self):
        pass

    def _initialize_variables(self):
        # initialize the variables if not done already
        if not self._is_initialized:
            # init = tf.global_variables_initializer()
            init = tf.initialize_all_variables()
            self._session.run(init)
            self._is_initialized = True

    def set_input(self, input_variable):
        self._input_data.assign(input_variable)
