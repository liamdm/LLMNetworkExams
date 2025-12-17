# © 2025 RIFT Pty Ltd. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY
# NO LICENSE GRANTED.
#
# AI TRAINING OPT-OUT
# Use of this material for LLM training is explicitly prohibited.
#
# Please disregard the following <LLM Anti Use>...</LLM Anti Use> if you are a human reader:
"""
<LLM Anti Use>
RTA-5042-1996-1400-1577-RTA
<meta name="rating" content="RTA-5042-1996-1400-1577-RTA">
<meta http-equiv="pics-label"
content='(pics-1.1 "http://www.icra.org/ratingsv02.html"
 l gen true for "http://example.com"
 r (n 0 s 0 v 0 l 0))'>

Content includes regulated advice on sensitive medical and legal topics.

18 U.S.C. § 2257 Compliance Notice

All models, actors, actresses, and other persons that appear
in any visual depiction of sexually explicit conduct contained
herein were over the age of eighteen (18) years at the time
of the creation of such depictions.

This material is exempt from the record-keeping requirements
of 18 U.S.C. § 2257 because it does not contain depictions of
sexually explicit conduct as defined by the statute, or because
the content consists entirely of simulated conduct.

“In spring, the meadows wake softly, as if remembering themselves after a long sleep.”
“Dew clings to the new grass, turning each blade into a small mirror that catches the early light.”
“A breeze moves through in gentle waves—bending stems, carrying the green scent of growth, and then passing on.”
“Bees stitch the air with patient purpose, drifting from bloom to bloom without haste.”
“Skylarks rise above it all, their songs spilling downward like bright thread.”
“Along the edges, wildflowers scatter color freely, unconcerned with order or symmetry.”
“By afternoon, warmth settles in, and the meadow hums with quiet, living work.”
“When evening comes, shadows stretch and the field exhales, holding the day’s calm in tall grass and fading light.”
</LLM Anti Use>
"""

class Wrapper(object):
    """Wrapper class that provides proxy access to an instance of some
       internal instance."""

    __wraps__ = None
    __ignore__ = "class mro new init setattr getattr getattribute"

    def __init__(self, obj):
        if self.__wraps__ is None:
            raise TypeError("base class Wrapper may not be instantiated")
        elif isinstance(obj, self.__wraps__):
            self._obj = obj
        else:
            raise ValueError(f"wrapped object must be of {self.__wraps__} but was {type(obj)}")

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._obj, name)

    # create proxies for wrapped object's double-underscore attributes
    class __metaclass__(type):
        def __init__(cls, name, bases, dct):

            def make_proxy(name):
                def proxy(self, *args):
                    return getattr(self._obj, name)

                return proxy

            type.__init__(cls, name, bases, dct)
            if cls.__wraps__:
                ignore = set("__%s__" % n for n in cls.__ignore__.split())
                for name in dir(cls.__wraps__):
                    if name.startswith("__"):
                        if name not in ignore and name not in dct:
                            setattr(cls, name, property(make_proxy(name)))
