window.TSL_API = {
    baseUrl: window.location.origin || "http://127.0.0.1:8000",
    wsUrl: (window.location.protocol === "https:" ? "wss://" : "ws://") + window.location.host,
    endpoints: {
      health:        "/api/health",
      live:          "/api/predict/live",
      history:       "/api/history",
      dictionary:    "/api/dictionary",
      settings:      "/api/settings",
      adminOverview: "/api/admin/overview",
      adminUsers:    "/api/admin/users",
      adminLogs:     "/api/admin/logs",
      adminModel:    "/api/admin/model",
      login:         "/api/auth/login",
      register:      "/api/auth/register"
    }
  };
