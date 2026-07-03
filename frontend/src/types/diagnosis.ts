// ── Request types ──────────────────────────────

export interface LabReport {
  wbc: number;
  neutrophil_pct: number;
  lymphocyte_pct: number;
  crp: number;
  temperature: number;
  systolic_bp: number;
  diastolic_bp: number;
  heart_rate: number;
  respiratory_rate: number;
  spo2: number;
  rbc: number;
  hemoglobin: number;
  hematocrit: number;
  platelet: number;
  glucose: number;
  creatinine: number;
  bun: number;
  alt: number;
  ast: number;
  total_cholesterol: number;
  triglycerides: number;
}

export interface DiagnoseRequest {
  name: string;
  age: number;
  gender: 'male' | 'female' | 'other';
  symptom_description: string;
  lab_report: LabReport;
}

// ── Response types ─────────────────────────────

export interface DiagnosisItem {
  disease: string;
  probability: number;
}

export interface DiagnosisResult {
  top_prediction: string;
  confidence: number;
  top3: DiagnosisItem[];
  treatment_suggestion: string;
}

export interface SimilarCase {
  id: number;
  similarity: number;
  symptom_description: string;
  diagnosis: string;
  treatment: string;
}

export interface DiagnoseResponse {
  diagnosis: DiagnosisResult;
  similar_cases: SimilarCase[];
}

export interface DiseaseInfo {
  name: string;
  description: string;
}

export interface PatientSummary {
  id: number;
  name: string;
  age: number;
  gender: string;
  diagnosis?: string;
  created_at?: string;
}

export interface PatientDetail extends PatientSummary {
  symptom_description: string;
  treatment?: string;
  lab_data?: Record<string, number>;
}
