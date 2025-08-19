import asyncio, json, websockets

HA_URL  = 'ws://localhost:8123/api/websocket'
TOKEN   = input('Enter your Home Assistant Long-Lived Access Token: ')

AREAS = [
    'Living Room',
    'Kitchen',
    'Dining Room',
    'Office',
    'Hallway',
    'Bedroom 1',
    'Bedroom 2',
    'Bedroom 3',
    'Bathroom',
    'Garage',
]

message_counter = 0

async def send_message(ws, msg_type, **extra):
    global message_counter
    message_counter += 1
    await ws.send(json.dumps({'id': message_counter, 'type': msg_type, **extra}))
    while True:
        resp = json.loads(await ws.recv())
        if resp.get('id') == message_counter:
            return resp

async def authenticate(ws):
    msg = json.loads(await ws.recv())
    assert msg['type'] == 'auth_required'
    await ws.send(json.dumps({'type': 'auth', 'access_token': TOKEN}))
    msg = json.loads(await ws.recv())
    assert msg['type'] == 'auth_ok'

async def create_areas(ws):
    response = await send_message(ws, 'config/area_registry/list')
    areas = response.get("result", [])

    for area in areas:
        await send_message(ws, 'config/area_registry/delete', area_id=area['area_id'])

    for area in AREAS:
        result = await send_message(ws, 'config/area_registry/create', name=area)
        print(f"Created area {area}: {result.get('error', 'Success')}")

async def assing_devices_to_areas(ws):
    areas = (await send_message(ws, 'config/area_registry/list')).get("result", [])
    area_by_name = {a['name']: a['area_id'] for a in areas}
    devices = (await send_message(ws, 'config/device_registry/list')).get("result", [])

    for device in devices:
        dev_name = device.get('name_by_user') or device.get('name') or ''
        for area_name in AREAS:
            if dev_name.startswith(area_name):
                area_id = area_by_name.get(area_name)
                if area_id:
                    print(f'Assigning {dev_name} → {area_name}')
                    await send_message(ws, 'config/device_registry/update', device_id=device['id'], area_id=area_id)
                    break

async def main():
    async with websockets.connect(HA_URL) as ws:
        print("\nAuthenticating ...")
        await authenticate(ws)
        print("\nCreating rooms ...")
        await create_areas(ws)
        print("\nAssigning devices to rooms ...")
        await assing_devices_to_areas(ws)       

asyncio.run(main())
