import type { Metadata } from "next";
import { readFileSync } from "node:fs";
import path from "node:path";

const globalStyles = readFileSync(path.join(process.cwd(), "app", "globals.css"), "utf-8");

export const metadata: Metadata = {
  title: "文本分析工作台",
  description: "中文文本分析平台",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <head>
        <style dangerouslySetInnerHTML={{ __html: globalStyles }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
