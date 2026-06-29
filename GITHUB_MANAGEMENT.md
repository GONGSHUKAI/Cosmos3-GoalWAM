# 完整操作步骤：本地NVIDIA原仓库推送代码到自己Fork仓库
## 一、先理清当前仓库远程状态
你现在本地仓库 `cosmos-framework` 默认远程 `origin` 指向 NVIDIA 官方仓库，**没有关联你的Fork仓库**，所以直接push会报错无权限。
思路：本地新增一个远程分支指向你的Fork，再把改动推过去。

## 1. 查看当前远程配置
进入本地仓库根目录执行：
```bash
git remote -v
```
输出大概是：
```
origin  https://github.com/NVIDIA/cosmos-framework.git (fetch)
origin  https://github.com/NVIDIA/cosmos-framework.git (push)
```

## 2. 添加你自己Fork的远程仓库（命名为myfork，可自定义）
```bash
git remote add myfork https://github.com/GONGSHUKAI/Cosmos3-GoalWAM.git
```
再次执行 `git remote -v` 验证，会多出两行myfork的拉取/推送地址。

## 3. 确认本地改动、提交代码（没commit必须先做）
### 查看修改文件
```bash
git status
```
### 全部暂存并提交（按需修改add范围）
```bash
git add .
git commit -m "GoalWAM 自定义开发改动，适配Cosmos3管线"
```

## 4. 推送到自己Fork仓库
假设你当前在主分支 `main` / `master`（替换成你实际分支名）：
```bash
# 推送到fork的同名分支
git push myfork main
```
如果本地分支和远端fork分支名不一致，格式：
```bash
git push myfork 本地分支名:远端分支名
```

# 补充高频场景解决方案
## 场景1：首次推送，远端无对应分支，提示upstream缺失
加 `-u` 绑定上下游，后续直接 `git push` 就能推myfork：
```bash
git push -u myfork main
```

## 场景2：后续NVIDIA官方仓库更新，同步上游代码到本地
```bash
# 拉取NVIDIA官方最新代码
git fetch origin
# 合并官方main到本地main
git merge origin/main
# 再推送到自己fork
git push myfork main
```

## 场景3：不想新建remote，直接把origin改成自己fork（不推荐，会丢失上游官方仓库关联）
```bash
git remote set-url origin https://github.com/GONGSHUKAI/Cosmos3-GoalWAM.git
```
缺点：之后无法一键拉取NVIDIA原版更新，**优先推荐上面新建myfork远程的方案**。

## 场景4：推送时报403无权限（https地址鉴权失败）
### 方案A：改用SSH地址推送（推荐）
1. 服务器配置GitHub ssh密钥
2. 修改myfork远程地址：
```bash
git remote set-url myfork git@github.com:GONGSHUKAI/Cosmos3-GoalWAM.git
```
再执行push即可免密码。

### 方案B：HTTPS使用个人访问令牌PAT
推送弹出账号密码时，密码输入GitHub生成的PAT令牌，不要填登录密码。

# 极简流程速记
```bash
# 1. 进入仓库
cd cosmos-framework
# 2. 绑定自己fork远程
git remote add myfork https://github.com/GONGSHUKAI/Cosmos3-GoalWAM.git
# 3. 提交改动
git add .
git commit -m "自定义GoalWAM开发更新"
# 4. 推送到个人fork
git push -u myfork main
```