from typing import Final

units: Final = {
    'mse': '\\si{(\\watt\\per\\square\\cm)\\squared}',
    'rmse': '\\si{\\watt\\per\\square\\cm}',
    'rms': '\\si{\\watt\\per\\square\\cm}',
    'mae': '\\si{\\watt\\per\\square\\cm}',
    'mape': '\\si{\\percent}',
    'r2': '---',
    'power': '\\si{\\watt}',
    'heat flux': '\\si{\\watt\\per\\square\\cm}',
    'temperature': '\\si{\\celsius}',
}

glossary: Final = {
    'power': 'q',
    'heat flux': 'q^{\\prime\\prime}',
    'excess temperature': '\\Delta T_{\\mathrm{e}}',
}
