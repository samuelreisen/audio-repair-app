# Generates clean reference speech with the built-in Windows voices (bench/clean/*.wav).
Add-Type -AssemblyName System.Speech
$out = Join-Path $PSScriptRoot 'clean'
New-Item -ItemType Directory -Force $out | Out-Null

$clips = @(
  @{ name = 'david_news';  voice = 'Microsoft David Desktop'; text = 'The committee met on Tuesday morning to discuss the new schedule. After a long debate, they agreed that the changes would begin next September, and that every family would receive a letter explaining the details.' },
  @{ name = 'zira_story';  voice = 'Microsoft Zira Desktop';  text = 'She opened the old wooden box slowly. Inside, wrapped in faded silk, was a small silver key, a photograph of a lighthouse, and a note that simply said: thank you for waiting so patiently.' },
  @{ name = 'david_sibil'; voice = 'Microsoft David Desktop'; text = 'Six slippery seals sat sunning on the sandy shore. Fresh fish and crisp chips were sold at the seaside shop, which opens at seven and closes precisely at six.' }
)
foreach ($c in $clips) {
  $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
  $s.SelectVoice($c.voice)
  $s.Rate = 0
  $s.SetOutputToWaveFile((Join-Path $out ($c.name + '.wav')))
  $s.Speak($c.text)
  $s.Dispose()
}
Get-ChildItem $out | Select-Object Name, Length
