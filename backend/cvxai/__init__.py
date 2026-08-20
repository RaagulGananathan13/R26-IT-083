"""
cvxai — unified backend for project R26-IT-083.

"Explainable AI System for Cardiovascular Disease Detection and Diagnosis".

Four independently-developed research components are exposed behind one FastAPI
service and one response contract:

    01  chest radiograph  -> cardiomegaly + 7 co-pathologies, Grad-CAM, report
    02  12-lead ECG       -> 5 superclasses, conformal triage, verified report
    03  echocardiogram    -> ejection fraction + 4-class severity grade
    04  ED triage record  -> ACS detection + UA/NSTEMI/STEMI subtyping

The components are NOT modified by this package. Their trained weights,
frozen decision rules and inference code are imported in place and driven
through thin adapters, so every number this service returns is the number the
component itself produces.
"""

__version__ = "1.0.0"
__project_id__ = "R26-IT-083"
