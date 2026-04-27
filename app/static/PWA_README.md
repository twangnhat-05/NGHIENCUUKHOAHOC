# 📱 PWA (Progressive Web App) Setup cho Streamlit

> Convert Streamlit dashboard → installable mobile/desktop app, hoạt động offline cache.

## Cách dùng

### Option A — Streamlit-PWA add-on (recommend)
```bash
pip install streamlit-pwa
```
Trong `app/streamlit_app.py` thêm đầu file:
```python
from streamlit_pwa import streamlit_pwa
streamlit_pwa(
    name="Gold SJC Forecast",
    short_name="Gold SJC",
    description="VN gold price forecasting",
    background_color="#ffffff",
    theme_color="#FFB800",
)
```

### Option B — Manual injection
Add vào `streamlit_app.py` ngay sau `st.set_page_config`:
```python
import streamlit.components.v1 as components
components.html("""
<link rel="manifest" href="/app/static/manifest.json">
<meta name="theme-color" content="#FFB800">
<link rel="apple-touch-icon" href="/app/static/icon-192.png">
""", height=0)
```

## Generate icons

Cần 2 icon PNG: `icon-192.png` (192x192) và `icon-512.png` (512x512).

Quick way (online):
- https://www.pwabuilder.com/imageGenerator → upload 1 PNG → tải zip có cả 2 sizes
- https://favicon.io/favicon-converter/

Place vào `app/static/`:
```
app/static/
├── manifest.json
├── icon-192.png
└── icon-512.png
```

## Deploy

Sau khi có manifest + icons:
- **Streamlit Cloud**: tự serve `app/static/*` files
- **Local**: `streamlit run app/streamlit_app.py` rồi mở Chrome → URL → Settings → "Install app"
- **Mobile**: open URL trên Chrome Android / Safari iOS → "Add to Home Screen"

## Limitations

- Streamlit không support service worker offline mode đầy đủ
- Cache strategy: chỉ static assets (manifest, icons) cache; app data cần online
- iOS Safari hỗ trợ "Add to Home Screen" nhưng UX hạn chế hơn Android Chrome

## Test PWA validity

Chrome DevTools → Application → Manifest → check:
- ✅ Name, short_name, icons present
- ✅ start_url correct
- ✅ display=standalone
- ✅ Service worker (optional — Streamlit không bundle)
