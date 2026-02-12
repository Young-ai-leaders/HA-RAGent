class DeviceExtractor:
    def __init__(self, entry, subentry_id):
        self.entry = entry
        self.subentry_id = subentry_id
        self.subentry = entry.subentries[subentry_id]


