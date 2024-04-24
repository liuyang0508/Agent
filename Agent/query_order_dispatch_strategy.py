import requests
import yaml


# DIQ 配置
SERVER_ADDR = 'http://nacos-dev.hzcxfw.com:8848'
NAMESPACE = 'itbox-nacos-dev'
DATA_ID = 'DIQ-matching-vehicle-service.yaml'
GROUP = 'DIQ'


def fetch_nacos_config(server_addr, namespace, data_id, group):
    # 构造请求URL
    url = f'{server_addr}/nacos/v1/cs/configs?dataId={data_id}&group={group}&tenant={namespace}'

    # 发送GET请求
    response = requests.get(url)

    # 检查响应状态码并处理结果
    if response.status_code == 200:
        config_content = response.text
        try:
            # 将内容解析为YAML格式
            yaml_config = yaml.safe_load(config_content)
            return yaml_config
        except yaml.YAMLError as exc:
            print(f"解析YAML配置失败：{exc}")
            return None
    else:
        print(f"获取配置失败，状态码：{response.status_code}")
        return None