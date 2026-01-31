"""
Node Management CLI for LARUN Federated TinyML System

Provides commands for managing analysis nodes:
- larun node list          - List all nodes and their status
- larun node enable <id>   - Enable a node for analysis
- larun node disable <id>  - Disable a node
- larun node install <id>  - Install a node
- larun node info <id>     - Show detailed node information
- larun node run <target> --nodes <ids>  - Run analysis with specific nodes

Usage:
    from src.cli.node_commands import NodeCommands

    cli = NodeCommands()
    cli.run(sys.argv[1:])
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


class NodeCommands:
    """CLI for LARUN node management."""

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize node commands CLI.

        Args:
            base_path: Base path for LARUN installation
        """
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent

        self.base_path = Path(base_path)

        # Lazy load to avoid import issues
        self._registry = None
        self._loader = None
        self._aggregator = None

    @property
    def registry(self):
        if self._registry is None:
            from nodes.registry import NodeRegistry
            self._registry = NodeRegistry(self.base_path)
        return self._registry

    @property
    def loader(self):
        if self._loader is None:
            from nodes.loader import NodeLoader
            self._loader = NodeLoader(self.registry)
        return self._loader

    @property
    def aggregator(self):
        if self._aggregator is None:
            from nodes.aggregator import NodeAggregator
            self._aggregator = NodeAggregator()
        return self._aggregator

    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for node commands."""
        parser = argparse.ArgumentParser(
            prog='larun node',
            description='LARUN Node Management CLI',
        )

        subparsers = parser.add_subparsers(dest='command', help='Node commands')

        # list command
        list_parser = subparsers.add_parser('list', help='List all nodes')
        list_parser.add_argument(
            '--category', '-c',
            help='Filter by category (exoplanet, stellar, transient, galactic, spectroscopy)'
        )
        list_parser.add_argument(
            '--status', '-s',
            choices=['available', 'installed', 'enabled', 'disabled'],
            help='Filter by status'
        )
        list_parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Show detailed information'
        )

        # enable command
        enable_parser = subparsers.add_parser('enable', help='Enable a node')
        enable_parser.add_argument('node_id', help='Node ID to enable (e.g., VSTAR-001)')

        # disable command
        disable_parser = subparsers.add_parser('disable', help='Disable a node')
        disable_parser.add_argument('node_id', help='Node ID to disable')

        # install command
        install_parser = subparsers.add_parser('install', help='Install a node')
        install_parser.add_argument(
            'node_id_or_url',
            help='Node ID (e.g., VSTAR-001) or URL to node package'
        )

        # uninstall command
        uninstall_parser = subparsers.add_parser('uninstall', help='Uninstall a node')
        uninstall_parser.add_argument('node_id', help='Node ID to uninstall')
        uninstall_parser.add_argument(
            '--keep-data',
            action='store_true',
            help='Keep node files but mark as uninstalled'
        )

        # info command
        info_parser = subparsers.add_parser('info', help='Show node information')
        info_parser.add_argument('node_id', help='Node ID to show info for')

        # run command
        run_parser = subparsers.add_parser('run', help='Run analysis with nodes')
        run_parser.add_argument('target', help='Target ID or file path')
        run_parser.add_argument(
            '--nodes', '-n',
            help='Comma-separated list of node IDs (default: enabled nodes)'
        )
        run_parser.add_argument(
            '--output', '-o',
            help='Output file for results (JSON)'
        )
        run_parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Show detailed output'
        )

        # stats command
        subparsers.add_parser('stats', help='Show node statistics')

        # validate command
        validate_parser = subparsers.add_parser('validate', help='Validate a node')
        validate_parser.add_argument('node_id', help='Node ID to validate')

        # refresh command
        subparsers.add_parser('refresh', help='Refresh node registry')

        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Run the CLI with given arguments.

        Args:
            args: Command line arguments (default: sys.argv[1:])

        Returns:
            Exit code (0 for success)
        """
        parser = self.create_parser()
        parsed = parser.parse_args(args)

        if parsed.command is None:
            parser.print_help()
            return 0

        # Dispatch to command handler
        handler = getattr(self, f'cmd_{parsed.command}', None)
        if handler is None:
            print(f"Unknown command: {parsed.command}")
            return 1

        try:
            return handler(parsed)
        except Exception as e:
            print(f"Error: {e}")
            return 1

    def cmd_list(self, args) -> int:
        """List nodes."""
        nodes = self.registry.list_nodes(
            category=args.category,
            status=args.status
        )

        print(self.registry.format_node_list(nodes, verbose=args.verbose))
        return 0

    def cmd_enable(self, args) -> int:
        """Enable a node."""
        success = self.registry.enable_node(args.node_id)
        return 0 if success else 1

    def cmd_disable(self, args) -> int:
        """Disable a node."""
        success = self.registry.disable_node(args.node_id)
        return 0 if success else 1

    def cmd_install(self, args) -> int:
        """Install a node."""
        success = self.registry.install_node(args.node_id_or_url)
        return 0 if success else 1

    def cmd_uninstall(self, args) -> int:
        """Uninstall a node."""
        success = self.registry.uninstall_node(
            args.node_id,
            keep_data=args.keep_data
        )
        return 0 if success else 1

    def cmd_info(self, args) -> int:
        """Show node information."""
        info = self.loader.get_node_info(args.node_id)

        if info is None:
            print(f"Node {args.node_id} not found")
            return 1

        print(f"\n Node: {info['node_id']}")
        print("=" * 50)
        print(f"  Name:        {info['name']}")
        print(f"  Version:     {info['version']}")
        print(f"  Category:    {info['category']}")
        print(f"  Status:      {info['status']}")
        print(f"  Model Size:  {info['model_size_kb']}KB")
        print(f"  Description: {info['description']}")

        if info.get('path'):
            print(f"  Path:        {info['path']}")

        if info.get('loaded'):
            print(f"  Loaded:      Yes")
            if info.get('input_shape'):
                print(f"  Input Shape: {info['input_shape']}")
            if info.get('output_classes'):
                print(f"  Classes:     {', '.join(info['output_classes'])}")

        return 0

    def cmd_run(self, args) -> int:
        """Run analysis with selected nodes."""
        # Determine which nodes to use
        if args.nodes:
            node_ids = [n.strip() for n in args.nodes.split(',')]
        else:
            # Use enabled nodes
            enabled = self.registry.get_enabled_nodes()
            node_ids = [n.node_id for n in enabled]

        if not node_ids:
            print("No nodes selected. Enable nodes or specify with --nodes")
            return 1

        print(f"\n Running {len(node_ids)} node(s): {', '.join(node_ids)}")
        print("-" * 50)

        # Load nodes
        nodes = self.loader.load_nodes(node_ids)

        if not nodes:
            print("No nodes could be loaded")
            return 1

        # Load or generate test data
        data = self._load_data(args.target)

        if data is None:
            print(f"Could not load data for target: {args.target}")
            return 1

        print(f"Loaded data: {data.shape}")

        # Run each node
        results = []
        for node in nodes:
            print(f"  Running {node.node_id}...", end=" ")
            result = node.run(data)
            results.append(result)

            if result.success:
                print(f"{result.classification} ({result.confidence:.1%}) "
                      f"[{result.inference_time_ms:.1f}ms]")
            else:
                print(f"FAILED: {result.error_message}")

        # Aggregate results
        aggregated = self.aggregator.aggregate(results, target_id=args.target)

        # Show results
        print(self.aggregator.format_result(aggregated, verbose=args.verbose))

        # Save output if requested
        if args.output:
            import json
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(aggregated.to_dict(), f, indent=2, default=str)
            print(f"\nResults saved to: {output_path}")

        return 0

    def cmd_stats(self, args) -> int:
        """Show node statistics."""
        stats = self.registry.get_stats()

        print("\n LARUN Node Statistics")
        print("=" * 40)
        print(f"  Total nodes:     {stats['total_nodes']}")
        print(f"  Available:       {stats['available']}")
        print(f"  Installed:       {stats['installed']}")
        print(f"  Enabled:         {stats['enabled']}")
        print(f"  Disabled:        {stats['disabled']}")
        print(f"  Total model KB:  {stats['total_model_size_kb']}KB")

        if stats['by_category']:
            print("\n  By Category:")
            for cat, count in sorted(stats['by_category'].items()):
                print(f"    {cat}: {count}")

        return 0

    def cmd_validate(self, args) -> int:
        """Validate a node."""
        print(f"Validating node: {args.node_id}")
        print("-" * 40)

        report = self.loader.validate_node(args.node_id)

        if report['valid']:
            print(" Node is valid")
        else:
            print(" Node has errors")

        if report['errors']:
            print("\nErrors:")
            for err in report['errors']:
                print(f"  - {err}")

        if report['warnings']:
            print("\nWarnings:")
            for warn in report['warnings']:
                print(f"  - {warn}")

        return 0 if report['valid'] else 1

    def cmd_refresh(self, args) -> int:
        """Refresh the node registry."""
        print("Refreshing node registry...")
        self.registry.refresh()
        print("Registry refreshed.")
        return 0

    def _load_data(self, target: str) -> Optional[np.ndarray]:
        """Load data for analysis."""
        # Check if it's a file path
        target_path = Path(target)
        if target_path.exists():
            if target_path.suffix == '.npy':
                return np.load(target_path)
            elif target_path.suffix == '.csv':
                return np.loadtxt(target_path, delimiter=',')
            elif target_path.suffix == '.txt':
                return np.loadtxt(target_path)
            else:
                # Try as numpy
                try:
                    return np.load(target_path, allow_pickle=True)
                except:
                    pass

        # Check if it's a TIC ID and try to fetch from MAST
        if target.upper().startswith('TIC') or target.isdigit():
            try:
                from lightkurve import search_lightcurve
                tic_id = target.replace('TIC', '').replace('_', '').strip()
                lc = search_lightcurve(f"TIC {tic_id}", mission="TESS")
                if lc:
                    lc_data = lc[0].download()
                    return lc_data.flux.value
            except ImportError:
                print("lightkurve not installed. Install with: pip install lightkurve")
            except Exception as e:
                print(f"Could not fetch TESS data: {e}")

        # Generate synthetic test data
        print(f"Generating synthetic test data for {target}")
        np.random.seed(hash(target) % 2**32)

        # Create a synthetic light curve with some features
        n_points = 1024
        t = np.linspace(0, 100, n_points)

        # Base flux with noise
        flux = 1.0 + 0.001 * np.random.randn(n_points)

        # Add a transit-like dip
        transit_center = 50
        transit_width = 2
        transit_depth = 0.01
        transit = transit_depth * np.exp(-(t - transit_center)**2 / (2 * transit_width**2))
        flux -= transit

        return flux


def main():
    """Main entry point."""
    cli = NodeCommands()
    sys.exit(cli.run())


if __name__ == '__main__':
    main()
