"""AutoBoat hardware subsystems.

Each module owns one device or OS facet: its config constants, its shared-state
dict + lock, and its polling loop. Every hardware import is guarded so the whole
package imports cleanly on a machine with none of the hardware (sections simply
report offline), the same property that makes the control package testable
anywhere. The dashboard imports these, wires the action registry, and starts the
loops; modules never import the dashboard.
"""
