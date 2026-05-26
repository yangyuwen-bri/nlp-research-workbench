"use client";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="zh-CN">
      <body
        style={{
          margin: 0,
          minHeight: "100vh",
          display: "grid",
          placeItems: "center",
          padding: 24,
          background: "#edf1f5",
          color: "#17212b",
          fontFamily: '"PingFang SC", "Microsoft YaHei", sans-serif',
        }}
      >
        <div
          style={{
            width: "min(560px, 100%)",
            padding: 24,
            border: "1px solid #d8e0e8",
            borderRadius: 16,
            background: "#ffffff",
          }}
        >
          <h2 style={{ margin: 0, fontSize: 24 }}>应用启动失败</h2>
          <p style={{ margin: "12px 0 0", color: "#667384", lineHeight: 1.6 }}>
            {error.message || "应用在初始化过程中出现异常。"}
          </p>
          <button
            onClick={() => reset()}
            style={{
              marginTop: 18,
              minHeight: 40,
              padding: "0 14px",
              borderRadius: 10,
              border: "1px solid #c5d0da",
              background: "#ffffff",
              color: "#17212b",
              cursor: "pointer",
            }}
            type="button"
          >
            重试
          </button>
        </div>
      </body>
    </html>
  );
}
