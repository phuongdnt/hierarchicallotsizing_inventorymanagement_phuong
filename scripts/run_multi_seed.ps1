$ErrorActionPreference = "Stop"

$seeds = 1..5
$packageRoot = $null

$rootCandidate = (Get-Location).Path
$scriptRoot = $PSScriptRoot
$scriptParent = $null

if ($scriptRoot) {
  $scriptParent = Split-Path -Parent $scriptRoot
}

if (Test-Path (Join-Path $rootCandidate "__init__.py") -and
    Test-Path (Join-Path $rootCandidate "train_main.py") -and
    Test-Path (Join-Path $rootCandidate "configs/train_serial.yaml")) {
  $packageRoot = $rootCandidate
} elseif (Test-Path (Join-Path $rootCandidate "hierarchicallotsizing_inventorymanagement_phuong/__init__.py") -and
          Test-Path (Join-Path $rootCandidate "hierarchicallotsizing_inventorymanagement_phuong/train_main.py") -and
          Test-Path (Join-Path $rootCandidate "hierarchicallotsizing_inventorymanagement_phuong/configs/train_serial.yaml")) {
  $packageRoot = (Join-Path $rootCandidate "hierarchicallotsizing_inventorymanagement_phuong")
} elseif ($scriptParent -and
          Test-Path (Join-Path $scriptParent "__init__.py") -and
          Test-Path (Join-Path $scriptParent "train_main.py") -and
          Test-Path (Join-Path $scriptParent "configs/train_serial.yaml")) {
  $packageRoot = $scriptParent
} elseif ($scriptParent -and
          Test-Path (Join-Path $scriptParent "hierarchicallotsizing_inventorymanagement_phuong/__init__.py") -and
          Test-Path (Join-Path $scriptParent "hierarchicallotsizing_inventorymanagement_phuong/train_main.py") -and
          Test-Path (Join-Path $scriptParent "hierarchicallotsizing_inventorymanagement_phuong/configs/train_serial.yaml")) {
  $packageRoot = (Join-Path $scriptParent "hierarchicallotsizing_inventorymanagement_phuong")
} else {
  $candidateDirs = Get-ChildItem -Path $rootCandidate -Directory -ErrorAction SilentlyContinue
  foreach ($dir in $candidateDirs) {
    $candidatePath = $dir.FullName
    if (Test-Path (Join-Path $candidatePath "__init__.py") -and
        Test-Path (Join-Path $candidatePath "train_main.py") -and
        Test-Path (Join-Path $candidatePath "configs/train_serial.yaml")) {
      $packageRoot = $candidatePath
      break
    }
  }
}

if (-not $packageRoot) {
  $searchRoot = $rootCandidate
  if ($scriptParent) {
    $searchRoot = $scriptParent
  }

  $configMatch = Get-ChildItem -Path $searchRoot -Recurse -Filter "train_serial.yaml" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match "configs[\\/]+train_serial.yaml$" } |
    Select-Object -First 1

  if ($configMatch) {
    $packageRoot = Split-Path -Parent (Split-Path -Parent $configMatch.FullName)
  }
}

if (-not $packageRoot) {
  throw "Could not locate package root containing train_main.py and configs/train_serial.yaml"
}

$baseConfigPath = (Join-Path $packageRoot "configs/train_serial.yaml")

if (-not (Test-Path $baseConfigPath)) {
  throw "Config file not found: $baseConfigPath"
}

$baseLines = Get-Content $baseConfigPath

if (-not ($baseLines | Select-String -Pattern '^\s*rng_seed:')) {
  $newLines = New-Object System.Collections.Generic.List[string]
  $inserted = $false
  foreach ($line in $baseLines) {
    $newLines.Add($line)
    if (-not $inserted -and $line -match '^\s*env:\s*$') {
      $newLines.Add("  rng_seed: 0")
      $inserted = $true
    }
  }

  if (-not $inserted) {
    $newLines.Add("env:")
    $newLines.Add("  rng_seed: 0")
  }

  $baseLines = $newLines.ToArray()
}

foreach ($s in $seeds) {
  $cfgPath = (Join-Path $packageRoot "configs/train_serial_seed$s.yaml")
  $configLines = $baseLines

  $configLines = $configLines -replace '^\s*rng_seed:.*', ("  rng_seed: {0}" -f $s)
  $configLines = $configLines -replace '^\s*save_path:.*', ('  save_path: "results/serial_model_seed{0}.pth"' -f $s)

  $configLines | Set-Content $cfgPath

  $runFrom = (Split-Path -Parent $packageRoot)
  if (-not $runFrom) {
    $runFrom = $packageRoot
  }

  Push-Location $runFrom
  try {
    python -m hierarchicallotsizing_inventorymanagement_phuong.train_main --config $cfgPath
  } finally {
    Pop-Location
  }
}
