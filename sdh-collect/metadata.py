def getUUID(data):
    if( 'uuid' in data ):
        uuid = data['uuid']
    else:
        uuid = '-'
    return uuid

def getUnit(data):
    if( 'UnitofMeasure' in data['Properties'] ):
        unit = data['Properties']['UnitofMeasure']
    else:
        unit = '-'
    return unit

def getPath(data):
    if "Path" in data:
        path = data['Path']
    else:
        path = '-'
    return path

def getSensortype(data):
    if( 'Instrument' in data['Metadata'] ):
        if( 'SensorType' in data['Metadata']['Instrument'] ):
            sensortype = data['Metadata']['Instrument']['SensorType']
        else:
            sensortype = '-'
    elif( 'SourceName' in data['Metadata'] ):
        sensortype = data['Metadata']['SourceName']
    return sensortype

def getNodeid(data):
    if( 'Instrument' in data['Metadata'] ):
        if( 'PartNumber' in data['Metadata']['Instrument'] ):
            nodeid = data['Metadata']['Instrument']['PartNumber']
        else:
            nodeid = '-'
    elif( 'Extra' in data['Metadata'] ):
        if( 'MeterName' in data['Metadata']['Extra'] ):
            nodeid = data['Metadata']['Extra']['MeterName']
        else:
            nodeid = "-"
    return nodeid

def getSerialNumber(data):
    if( 'SerialNumber' in data['Metadata'] ):
        serialnumber = data['Metadata']['SerialNumber']
    else:
        serialnumber = "-"
    return serialnumber

def getLocationBD(data):
    if( 'Location' in data['Metadata'] ):
        if( 'Building' in data['Metadata']['Location'] ):
            location = data['Metadata']['Location']['Building']
        else:
            location ='-'
    else:
        location ='-'
    return location

def getLocationFL(data):
    if( 'Location' in data['Metadata'] ):
        if( 'Floor' in data['Metadata']['Location'] ):
            location = data['Metadata']['Location']['Floor'] + "FL"
        else:
            location = "-"
    elif( 'PointName' in data['Metadata'] ):
        location = data['Metadata']['PointName']
    else:
        location = "-"
    return location

def getLocationRM(data):
    if( 'Location' in data['Metadata'] ):
        if( 'RoomNumber' in data['Metadata']['Location'] ):
            location = data['Metadata']['Location']['RoomNumber']
        else:
            location ='-'
    elif( 'Extra' in data['Metadata'] ):
        if( 'ServiceArea' in data['Metadata']['Extra'] ):
            location = data['Metadata']['Extra']['ServiceArea']
        else:
            location = "-"
    elif( 'Description' in data ):
        location = data['Description']
    else:
        location = "-"
    return location

def getInstalledInfo(data):
    if( 'Location' in data['Metadata'] ):
        if( 'Height' in data['Metadata']['Location'] ):
            installedInfo = data['Metadata']['Location']['Height']
        else:
            installedInfo = '-'
    elif( 'Extra' in data['Metadata'] ):
        if( 'System' in data['Metadata']['Extra'] ):
            installedInfo = data['Metadata']['Extra']['System']
        elif( 'Operator' in data['Metadata']['Extra'] ):
            installedInfo = data['Metadata']['Extra']['Operator']
        else:
            installedInfo = '-'
    else:
        installedInfo = '-'
    return installedInfo

def getAddtionalInfo(data):
    if( 'Extra' in data['Metadata'] ):
        if( 'VAV' in data['Metadata']['Extra'] ):
            addtionalInfo = data['Metadata']['Extra']['VAV']
        elif( 'ServiceDetail' in data['Metadata']['Extra'] ):
            addtionalInfo = data['Metadata']['Extra']['ServiceDetail']
        elif( 'Type' in data['Metadata']['Extra'] ):
            addtionalInfo = data['Metadata']['Extra']['Type']
        else:
            addtionalInfo = '-'
    else:
        addtionalInfo = '-'
    return addtionalInfo
