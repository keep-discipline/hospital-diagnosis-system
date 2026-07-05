import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import type { PatientSummary } from '../types/diagnosis';

const PAGE_SIZE = 20;

export default function HistoryPage() {
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<PatientSummary | null>(null);

  const fetchPatients = useCallback(async (p: number, q: string) => {
    setLoading(true);
    try {
      const { data } = await api.getPatients(p * PAGE_SIZE, PAGE_SIZE, q);
      setPatients(data.data);
      setTotal(data.total);
    } catch {
      setPatients([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPatients(page, search);
  }, [page, search, fetchPatients]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(0);
    fetchPatients(0, search);
  };

  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <div className="history-page">
      {/* 搜索栏 */}
      <form className="history-search" onSubmit={handleSearch}>
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="搜索姓名、症状或诊断..."
          className="search-input"
        />
        <button type="submit" className="btn btn-primary btn-sm">
          🔍 搜索
        </button>
        {search && (
          <button
            type="button"
            className="btn btn-outline btn-sm"
            onClick={() => { setSearch(''); setPage(0); }}
          >
            清除
          </button>
        )}
      </form>

      <div className="history-stats">
        {loading ? '加载中...' : `共 ${total} 条记录`}
      </div>

      {/* 列表 */}
      {loading ? (
        <div className="card"><div className="skeleton-line skeleton-text" /></div>
      ) : patients.length === 0 ? (
        <div className="card empty-hint">
          {search ? '没有找到匹配的记录' : '暂无诊断记录，去首页做一个诊断吧'}
        </div>
      ) : (
        <div className="history-list">
          {patients.map((p) => (
            <div
              key={p.id}
              className={`history-item ${selected?.id === p.id ? 'selected' : ''}`}
              onClick={() => setSelected(selected?.id === p.id ? null : p)}
            >
              <div className="history-item-main">
                <span className="history-name">{p.name}</span>
                <span className="history-meta">{p.age}岁 · {p.gender === 'male' ? '男' : p.gender === 'female' ? '女' : '其他'}</span>
                <span className="history-diagnosis">{p.diagnosis || '待确认'}</span>
              </div>
              <span className="history-date">
                {p.created_at ? new Date(p.created_at).toLocaleDateString('zh-CN') : '-'}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* 分页 */}
      {totalPages > 1 && (
        <div className="pagination">
          <button disabled={page === 0} onClick={() => setPage(page - 1)}>◀ 上一页</button>
          <span>{page + 1} / {totalPages}</span>
          <button disabled={page >= totalPages - 1} onClick={() => setPage(page + 1)}>下一页 ▶</button>
        </div>
      )}
    </div>
  );
}
