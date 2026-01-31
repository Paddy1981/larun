#
# LARUN TinyML - Installation Script for Windows
# ===============================================
# One-command installation for Windows PowerShell
#
# Usage:
#   irm https://raw.githubusercontent.com/Paddy1981/larun/main/install.ps1 | iex
#   OR
#   .\install.ps1
#

$ErrorActionPreference = "Stop"

# Colors
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     ██╗      █████╗ ██████╗ ██╗   ██╗███╗   ██╗                      ║" -ForegroundColor Cyan
Write-Host "║     ██║     ██╔══██╗██╔══██╗██║   ██║████╗  ██║                      ║" -ForegroundColor Cyan
Write-Host "║     ██║     ███████║██████╔╝██║   ██║██╔██╗ ██║                      ║" -ForegroundColor Cyan
Write-Host "║     ██║     ██╔══██║██╔══██╗██║   ██║██║╚██╗██║                      ║" -ForegroundColor Cyan
Write-Host "║     ███████╗██║  ██║██║  ██║╚██████╔╝██║ ╚████║                      ║" -ForegroundColor Cyan
Write-Host "║     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝                      ║" -ForegroundColor Cyan
Write-Host "║                                                                      ║" -ForegroundColor Cyan
Write-Host "║     TinyML for Astronomical Data Analysis                            ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "Checking prerequisites..." -ForegroundColor Cyan

function Test-PythonVersion {
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python (\d+)\.(\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            
            if ($major -ge 3 -and $minor -ge 8) {
                Write-Host "✓ $pythonVersion" -ForegroundColor Green
                return $true
            } else {
                Write-Host "✗ Python $major.$minor found, but Python 3.8+ is required" -ForegroundColor Red
                return $false
            }
        }
    } catch {
        Write-Host "✗ Python not found!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please install Python 3.8 or newer:"
        Write-Host "  1. Download from https://www.python.org/downloads/"
        Write-Host "  2. Run the installer"
        Write-Host "  3. IMPORTANT: Check 'Add Python to PATH' during installation"
        Write-Host ""
        return $false
    }
}

function Test-Pip {
    try {
        $pipVersion = python -m pip --version 2>&1
        if ($pipVersion -match "pip (\d+\.\d+)") {
            Write-Host "✓ pip $($Matches[1])" -ForegroundColor Green
            return $true
        }
    } catch {
        Write-Host "✗ pip not found!" -ForegroundColor Red
        return $false
    }
}

function Test-Git {
    try {
        $gitVersion = git --version 2>&1
        if ($gitVersion -match "git version (\d+\.\d+)") {
            Write-Host "✓ git $($Matches[1])" -ForegroundColor Green
            return $true
        }
    } catch {
        Write-Host "⚠ git not found (optional, for development)" -ForegroundColor Yellow
        return $true
    }
}

# Run checks
if (-not (Test-PythonVersion)) {
    exit 1
}

if (-not (Test-Pip)) {
    Write-Host "Installing pip..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "get-pip.py"
    python get-pip.py
    Remove-Item "get-pip.py"
}

Test-Git | Out-Null

Write-Host ""

# Installation options
Write-Host "Choose installation method:" -ForegroundColor Cyan
Write-Host "  1) Install from PyPI (recommended)"
Write-Host "  2) Install from GitHub (latest)"
Write-Host "  3) Install in development mode (for contributors)"
Write-Host ""
$choice = Read-Host "Enter choice [1]"
if ([string]::IsNullOrEmpty($choice)) { $choice = "1" }

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "Installing from PyPI..." -ForegroundColor Cyan
        python -m pip install --upgrade larun
    }
    "2" {
        Write-Host ""
        Write-Host "Installing from GitHub..." -ForegroundColor Cyan
        python -m pip install --upgrade "git+https://github.com/Paddy1981/larun.git"
    }
    "3" {
        Write-Host ""
        Write-Host "Cloning repository..." -ForegroundColor Cyan
        if (Test-Path "larun") {
            Write-Host "Directory 'larun' already exists. Using existing directory."
            Set-Location "larun"
            git pull 2>$null
        } else {
            git clone https://github.com/Paddy1981/larun.git
            Set-Location "larun"
        }
        Write-Host ""
        Write-Host "Installing in development mode..." -ForegroundColor Cyan
        python -m pip install -e ".[dev]"
    }
    default {
        Write-Host "Invalid choice" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Verify installation
Write-Host "Verifying installation..." -ForegroundColor Cyan

try {
    $larunCheck = Get-Command larun -ErrorAction SilentlyContinue
    if ($larunCheck) {
        Write-Host "✓ larun command is available" -ForegroundColor Green
    } else {
        Write-Host "⚠ larun command not found in PATH" -ForegroundColor Yellow
        Write-Host "  You may need to restart your terminal or add Python Scripts to PATH"
    }
} catch {
    Write-Host "⚠ larun command not found in PATH" -ForegroundColor Yellow
}

try {
    $larunChatCheck = Get-Command larun-chat -ErrorAction SilentlyContinue
    if ($larunChatCheck) {
        Write-Host "✓ larun-chat command is available" -ForegroundColor Green
    }
} catch {
    # Silently ignore
}

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                    Installation Complete!                            ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Quick start:"
Write-Host "  larun          # Start interactive CLI"
Write-Host "  larun-chat     # Start chat interface"
Write-Host ""
Write-Host "Documentation: https://github.com/Paddy1981/larun"
Write-Host ""
