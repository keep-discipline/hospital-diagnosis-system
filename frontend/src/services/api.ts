import axios from 'axios';
import type {
  DiagnoseRequest,
  DiagnoseResponse,
  DiseaseInfo,
  PatientSummary,
  PatientDetail,
} from '../types/diagnosis';

const client = axios.create({
  baseURL: '/api',
  timeout: 60000, // 模型推理可能需要一些时间
  headers: { 'Content-Type': 'application/json' },
});

export const api = {
  /** 提交诊断请求 */
  diagnose(data: DiagnoseRequest): Promise<{ data: DiagnoseResponse }> {
    return client.post('/diagnose', data);
  },

  /** 获取支持的疾病列表 */
  getDiseases(): Promise<{ data: DiseaseInfo[] }> {
    return client.get('/diseases');
  },

  /** 分页查询病人，支持搜索 */
  getPatients(
    skip = 0,
    limit = 20,
    q = ''
  ): Promise<{ data: { total: number; data: PatientSummary[] } }> {
    return client.get('/patients', { params: { skip, limit, q } });
  },

  /** 查询单个病人详情 */
  getPatient(id: number): Promise<{ data: PatientDetail }> {
    return client.get(`/patients/${id}`);
  },

  /** 健康检查 */
  healthCheck(): Promise<{ data: { status: string } }> {
    return client.get('/health');
  },
};
