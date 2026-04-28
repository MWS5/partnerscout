# PartnerScout AI — Admin Guide (Owner Only)

> Для отладки, тестирования и мониторинга без Stripe и без лимитов.

---

## 🔑 Admin URL

```
https://jares-ai.com/partnerscout?admin=ps_admin_mws5_2026
```

Параметр `?admin=SECRET` активирует Admin Mode:
- ✅ Нет Stripe — оплата не требуется
- ✅ Полные данные — email, контакты без blur
- ✅ 50 компаний вместо 10
- ✅ Dashboard показывает "⚡ Admin: full results ready!"

---

## 🧪 Сценарий тестирования (шаг за шагом)

### 1. Открыть форму
```
https://jares-ai.com/partnerscout?admin=ps_admin_mws5_2026
```

### 2. Заполнить форму
- **Email:** любой (например mwebstudio5@gmail.com)
- **Niche:** hotel / event_agency / wedding / concierge / travel / venue
- **Region:** любой (например "Côte d'Azur, France")
- **Segment:** luxury
- Нажать "Get 10 free leads →"

### 3. Редирект на Dashboard
После отправки → автоматически переходит на:
```
https://jares-ai.com/partnerscout/dashboard?order_id=UUID&admin=ps_admin_mws5_2026
```
Прогресс-бар обновляется каждые 3 сек.

### 4. Ждать результатов (2–5 минут)
Пайплайн: Discovery → Extraction → Enrichment → Output

### 5. Результаты
- Таблица: Company / Category / Email / Contact / Score
- Email и контакты — **не блюрены** (admin привилегия)
- Заголовок: "⚡ Admin: full results ready! — Full unblurred data — 50 companies"

---

## 🔌 API — прямые запросы для отладки

### Health check
```bash
curl https://partnerscout-api-production.up.railway.app/ping
# → {"status":"pong","service":"partnerscout"}
```

### Создать admin order напрямую
```bash
curl -X POST https://partnerscout-api-production.up.railway.app/api/v1/orders/admin \
  -H "Content-Type: application/json" \
  -H "X-Admin-Secret: ps_admin_mws5_2026" \
  -d '{
    "email": "mwebstudio5@gmail.com",
    "niches": ["hotel", "wedding"],
    "regions": ["Côte d'\''Azur, France"],
    "segment": "luxury",
    "count_target": 50
  }'
# → {"order_id": "UUID", "status": "running", "poll_url": "/api/v1/orders/UUID"}
```

### Проверить статус заказа
```bash
curl https://partnerscout-api-production.up.railway.app/api/v1/orders/UUID
# → {"status": "running", "progress": 45, ...}
# → {"status": "done", "progress": 100, ...}
```

### Скачать результаты (JSON)
```bash
curl https://partnerscout-api-production.up.railway.app/api/v1/export/UUID/json \
  -H "X-Admin-Secret: ps_admin_mws5_2026"
```

### Скачать результаты (CSV)
```bash
curl https://partnerscout-api-production.up.railway.app/api/v1/export/UUID/csv \
  -H "X-Admin-Secret: ps_admin_mws5_2026" -o results.csv
```

---

## 🚀 Деплой обновлений

```bash
cd C:\Users\mwebs\Downloads\ai_business\workspaces\partnerscout
RAILWAY_TOKEN="$RAILWAY_TOKEN"  # set via shell env (NEVER commit token here) railway up --service "1293d163-605f-40d6-8a1b-a0672b7282d2" --detach
```

---

## 📊 Supabase — прямой доступ к данным

- **Dashboard:** https://supabase.com/dashboard/project/jdvivkzggloetuakbqky
- **Таблицы:**
  - `ps_orders` — все заказы (статус, прогресс, is_trial)
  - `ps_results` — компании по каждому заказу (luxury_score, email, contact)
  - `ps_search_cache` — кеш поисковых результатов

### SQL для мониторинга
```sql
-- Последние 10 заказов
SELECT id, email, status, progress, is_trial, created_at 
FROM ps_orders ORDER BY created_at DESC LIMIT 10;

-- Результаты конкретного заказа
SELECT company_name, luxury_score, email, contact_person, category
FROM ps_results WHERE order_id = 'UUID' ORDER BY luxury_score DESC;

-- Статистика
SELECT status, COUNT(*) FROM ps_orders GROUP BY status;
```

---

## ⚙️ Railway ENV vars (установлены)

| Переменная | Статус |
|---|---|
| OPENROUTER_API_KEY | ✅ |
| BRAVE_API_KEY | ✅ |
| SEARXNG_URL | ✅ (internal Railway) |
| DATABASE_URL | ✅ Supabase |
| SUPABASE_URL / SUPABASE_KEY | ✅ |
| JARVIS_WEBHOOK_URL | ✅ |
| ADMIN_SECRET | ✅ ps_admin_mws5_2026 |

---

## 🔍 Логи Railway

```bash
RAILWAY_TOKEN="$RAILWAY_TOKEN"  # set via shell env (NEVER commit token here) railway logs --service "1293d163-605f-40d6-8a1b-a0672b7282d2"
```

Или в браузере: https://railway.app/project/7e7ebe3b-913a-4790-9989-b7320318fb29
