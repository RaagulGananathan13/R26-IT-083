// Synthetic illustrations, NOT machine-learning output and NOT clinical advice.
// The percentages below are invented ONLY to demonstrate the interface.
export const scenarios = [
  {
    id: 'case-01', title: 'Anterior STEMI', subtitle: 'Illustrative high-acuity presentation',
    patient: { age: 61, sex: 'M', heartrate: 108, sbp: 92, dbp: 58, o2sat: 93, resprate: 24, pain: 9, acuity: 1, chief_complaint: 'Crushing chest pain radiating to left arm, sweating', troponin_first: 1.2, troponin_second: 6.8, st_elevation: true, st_depression: false, t_inversion: false },
    result: { prediction: 'STEMI', p_acs: .98, probabilities: { No_ACS: .02, UA: .02, NSTEMI: .10, STEMI: .86 }, risk_level: 'High', recommended_action: 'Demonstration only — clinician assessment required.', feature_importance: [{ feature: 'ST elevation', value: .34 }, { feature: 'Troponin change', value: .26 }, { feature: 'Chest pain', value: .18 }, { feature: 'Low systolic BP', value: .13 }, { feature: 'Heart rate', value: .09 }], modality_attribution: { ECG: .39, Labs: .28, Text: .19, Vitals: .14 }, text_attribution: [{ term: 'chest pain', category: 'symptom', negated: false }, { term: 'left arm', category: 'radiation', negated: false }, { term: 'sweating', category: 'associated symptom', negated: false }] }
  },
  {
    id: 'case-02', title: 'NSTEMI', subtitle: 'Illustrative biomarker-positive case',
    patient: { age: 74, sex: 'F', heartrate: 92, sbp: 138, dbp: 80, o2sat: 96, resprate: 20, pain: 6, acuity: 2, chief_complaint: 'Chest pressure and shortness of breath', troponin_first: .28, troponin_second: .51, st_elevation: false, st_depression: true, t_inversion: true },
    result: { prediction: 'NSTEMI', p_acs: .88, probabilities: { No_ACS: .12, UA: .05, NSTEMI: .76, STEMI: .07 }, risk_level: 'High', recommended_action: 'Demonstration only — clinician assessment required.', feature_importance: [{ feature: 'Troponin change', value: .37 }, { feature: 'ST depression', value: .24 }, { feature: 'T-wave inversion', value: .17 }, { feature: 'Age', value: .13 }, { feature: 'Complaint text', value: .09 }], modality_attribution: { Labs: .41, ECG: .32, Text: .15, Vitals: .12 }, text_attribution: [{ term: 'chest pressure', category: 'symptom', negated: false }, { term: 'shortness of breath', category: 'symptom', negated: false }] }
  },
  {
    id: 'case-03', title: 'Unstable angina', subtitle: 'Illustrative normal-biomarker presentation',
    patient: { age: 58, sex: 'M', heartrate: 78, sbp: 146, dbp: 88, o2sat: 98, resprate: 16, pain: 5, acuity: 2, chief_complaint: 'Chest pain on exertion, resolves at rest', troponin_first: .01, troponin_second: '', st_elevation: false, st_depression: false, t_inversion: false },
    result: { prediction: 'UA', p_acs: .70, probabilities: { No_ACS: .30, UA: .55, NSTEMI: .11, STEMI: .04 }, risk_level: 'Elevated', recommended_action: 'Demonstration only — clinician assessment required.', feature_importance: [{ feature: 'Exertional symptoms', value: .33 }, { feature: 'Complaint text', value: .27 }, { feature: 'Age', value: .16 }, { feature: 'Pain', value: .14 }, { feature: 'ECG interpretation', value: .10 }], modality_attribution: { Text: .43, Vitals: .21, ECG: .18, Labs: .18 }, text_attribution: [{ term: 'chest pain', category: 'symptom', negated: false }, { term: 'exertion', category: 'trigger', negated: false }] }
  },
  {
    id: 'case-04', title: 'No ACS', subtitle: 'Illustrative non-cardiac complaint',
    patient: { age: 34, sex: 'F', heartrate: 84, sbp: 118, dbp: 74, o2sat: 99, resprate: 16, pain: 4, acuity: 3, chief_complaint: 'Abdominal pain and nausea, denies chest pain', troponin_first: '', troponin_second: '', st_elevation: false, st_depression: false, t_inversion: false },
    result: { prediction: 'No_ACS', p_acs: .08, probabilities: { No_ACS: .92, UA: .03, NSTEMI: .03, STEMI: .02 }, risk_level: 'Undetermined', recommended_action: 'Demonstration only — normal result cannot rule out disease.', feature_importance: [{ feature: 'Chief complaint', value: .38 }, { feature: 'Age', value: .21 }, { feature: 'Vital signs', value: .18 }, { feature: 'No recorded ECG', value: .13 }, { feature: 'No recorded troponin', value: .10 }], modality_attribution: { Text: .42, Vitals: .31, Labs: .14, ECG: .13 }, text_attribution: [{ term: 'abdominal pain', category: 'symptom', negated: false }, { term: 'chest pain', category: 'negated symptom', negated: true }] }
  }
]
