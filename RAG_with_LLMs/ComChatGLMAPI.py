import datetime
import json
import uvicorn
import zhipuai
from fastapi import FastAPI, Request

app = FastAPI()


# 选择的是https://open.bigmodel.cn/dev/api智谱清言的大模型API 可以换成本地部署的大模型api或其他平台的大模型api
@app.post("/")
async def create_item(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")

    # -------------------------- 替换的API调用部分 --------------------------
    # 创建ZhipuAI客户端实例（替换原zhipuai.api_key直接赋值）
    client = zhipuai.ZhipuAI(api_key="ec6e2c64991f4b6eb7a053576bd0aaa5.uhxv7j6NTIaeO3Aq")
    # 使用client.chat.completions.create调用模型（替换原model_api.invoke）
    response = client.chat.completions.create(
        model="glm-4.6",  # 模型名称更新为glm-4.6
        messages=[
            {"role": "system", "content": "你是一个有用的AI助手。"},  # 新增system角色提示
            {"role": "user", "content": prompt}  # user角色内容使用变量prompt
        ],
        temperature=0.7  # 保留原温度参数
    )
    # 从新响应结构提取回答（替换原response['data']['choices'][0]['content']）
    RESPONSE = response.choices[0].message.content.replace("\\n", "\n")
    # -------------------------- 替换结束 --------------------------

    answer = {
        "response": RESPONSE,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    return answer


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)