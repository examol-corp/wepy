class Missing:
    # no data allowed
    __slots__ = ()

    def __repr__(self):
        return "<Missing>"


# Single instance
MISSING = Missing()
