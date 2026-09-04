"""Phasor-domain (positive-sequence) dynamic layer, built on ANDES.

This is the campaign *workhorse*: cheap enough to bisect on, detailed enough to
have real GFM droop dynamics, current-loop states and a synchronous machine.
It is not the ground truth -- EMT (plan Part 0-D1) is. Every boundary produced
here carries `platform = "andes"` in its artifact so the EMT cross-check at T5/T6
can be joined against it row by row.
"""
