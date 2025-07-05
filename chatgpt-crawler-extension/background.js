chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    fetch('https://tradelogsai.eastus.cloudapp.azure.com/register', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
      //,body: JSON.stringify({ timestamp: new Date().toISOString() })
    })
    .then(res => res.json())
    .then(data => {
      if (data.machineId) {
        chrome.storage.local.set({ machineId: data.machineId }, () => {
          console.log('machineId saved:', data.machineId);
        });
      }
    })
    .catch(err => console.error('Install tracking failed:', err));
  }
});