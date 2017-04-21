# -*- coding: utf-8 -*-


"""Declare errors for ngface.

"""

# --- error message ---
class NgException(Exception):
    def __init__(self, code, msg):
        self.code = code
        self.msg = msg


class MissArgsError(NgException):
    """Miss arguments error.
    """

    def __init__(self, msg):
        super(MissArgsError, self).__init__(101, msg)

    def __str__(self):
        return 'Miss args error. {}'.format(self.msg)
