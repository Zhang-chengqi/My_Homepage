# 张诚琪的个人主页

基于 Jekyll + GitHub Pages 构建的个人主页与博客。

## 快速开始

1. 本地预览（需要安装 Jekyll）：
   ```bash
   bundle install
   bundle exec jekyll serve
   ```

2. 访问 https://zhang-chengqi.github.io/My_Homepage/ 预览

## 目录结构

- `index.html` - 主页
- `blog.html` - 博客列表页
- `_posts/` - 博客文章目录
- `_layouts/` - 页面布局模板
- `assets/css/style.css` - 样式文件

## 添加新博客

在 `_posts/` 目录下创建新文件，文件名格式：`年-月-日-标题.md`

文件开头需要包含如下 front matter：
```yaml
---
layout: post
title: 你的标题
date: 2026-05-10
description: 简短描述
categories:
  - 分类
---
```

## 部署

推送到 GitHub 后，在仓库设置中启用 GitHub Pages，Source 选择 "main branch"。