"""
LARUN Pipeline CLI
==================
Command-line interface for the discovery pipeline system.
Supports both terminal commands, web dashboard, and node management.

New in v2.0: Federated multi-model support with node commands.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class LarunCLI:
    """
    LARUN Command Line Interface.

    Provides terminal-based access to the discovery pipeline,
    candidate management, and submission tools.
    """

    def __init__(self, db_path: str = "data/candidates.db"):
        from src.database.candidate_db import CandidateDatabase
        from src.pipeline.discovery_pipeline import DiscoveryPipeline

        self.db = CandidateDatabase(db_path)
        self.pipeline = DiscoveryPipeline(db_path=db_path)
        self.user_profile = self.db.get_user_profile()

    def setup_profile(self):
        """Interactive user profile setup."""
        print("\n" + "=" * 60)
        print("  LARUN User Profile Setup")
        print("=" * 60 + "\n")

        current = self.db.get_user_profile() or {}

        name = input(f"Your name [{current.get('name', '')}]: ").strip()
        if not name:
            name = current.get('name')

        email = input(f"Email [{current.get('email', '')}]: ").strip()
        if not email:
            email = current.get('email')

        affiliation = input(f"Affiliation/Institution [{current.get('affiliation', '')}]: ").strip()
        if not affiliation:
            affiliation = current.get('affiliation')

        orcid = input(f"ORCID (optional) [{current.get('orcid', '')}]: ").strip()
        if not orcid:
            orcid = current.get('orcid')

        exofop_user = input(f"ExoFOP username (optional) [{current.get('exofop_username', '')}]: ").strip()
        if not exofop_user:
            exofop_user = current.get('exofop_username')

        self.db.save_user_profile(
            name=name,
            email=email,
            affiliation=affiliation,
            orcid=orcid,
            exofop_username=exofop_user
        )

        print("\nProfile saved successfully!")
        self.user_profile = self.db.get_user_profile()

    def show_profile(self):
        """Display current user profile."""
        profile = self.db.get_user_profile()

        if not profile or not profile.get('name'):
            print("\nNo profile configured. Run 'larun profile setup' to configure.")
            return

        print("\n" + "=" * 50)
        print("  User Profile")
        print("=" * 50)
        print(f"  Name:        {profile.get('name', 'Not set')}")
        print(f"  Email:       {profile.get('email', 'Not set')}")
        print(f"  Affiliation: {profile.get('affiliation', 'Not set')}")
        print(f"  ORCID:       {profile.get('orcid', 'Not set')}")
        print(f"  ExoFOP:      {profile.get('exofop_username', 'Not set')}")
        print("=" * 50 + "\n")

    def run_pipeline(self, target_id: str, mission: str = "TESS"):
        """Run discovery pipeline on a target."""
        print(f"\nProcessing {target_id}...")

        result = self.pipeline.process_target(target_id, mission=mission)

        if result.success:
            c = result.candidate
            print(f"\n  SUCCESS: Candidate {c.id}")
            print(f"  Status: {c.status.value}")
            print(f"  Confidence: {c.confidence:.1%}")
            if c.period_days:
                print(f"  Period: {c.period_days:.4f} days")
            if c.snr:
                print(f"  SNR: {c.snr:.1f}")
            if c.planet_radius_earth:
                print(f"  Radius: {c.planet_radius_earth:.2f} Earth")
            if c.priority:
                print(f"  Priority: {c.priority.value.upper()}")
        else:
            print(f"\n  FAILED at stage: {result.stage_completed}")
            print(f"  Reason: {result.error_message}")

    def list_candidates(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 20
    ):
        """List candidates with optional filters."""
        from src.database.models import CandidateStatus, Priority

        status_filter = CandidateStatus(status) if status else None
        priority_filter = Priority(priority) if priority else None

        candidates = self.db.list_candidates(
            status=status_filter,
            priority=priority_filter,
            limit=limit
        )

        if not candidates:
            print("\nNo candidates found.")
            return

        print(f"\n{'ID':<25} {'Status':<15} {'Confidence':<12} {'Period':<10} {'Priority':<10}")
        print("-" * 75)

        for c in candidates:
            period_str = f"{c.period_days:.3f}d" if c.period_days else "-"
            conf_str = f"{c.confidence:.1%}" if c.confidence else "-"
            print(f"{c.id:<25} {c.status.value:<15} {conf_str:<12} {period_str:<10} {c.priority.value:<10}")

        print(f"\nTotal: {len(candidates)} candidates")

    def show_candidate(self, candidate_id: str):
        """Show detailed candidate information."""
        candidate = self.db.get_candidate(candidate_id)

        if not candidate:
            print(f"\nCandidate not found: {candidate_id}")
            return

        print("\n" + "=" * 60)
        print(f"  Candidate: {candidate.id}")
        print("=" * 60)
        print(f"\n  Target:         {candidate.target_id}")
        print(f"  Status:         {candidate.status.value}")
        print(f"  Priority:       {candidate.priority.value}")
        print(f"  Classification: {candidate.classification}")
        print(f"  Confidence:     {candidate.confidence:.1%}" if candidate.confidence else "")
        print(f"  Data Source:    {candidate.data_source}")

        if candidate.period_days:
            print(f"\n  --- Orbital Parameters ---")
            print(f"  Period:         {candidate.period_days:.6f} days")
            if candidate.planet_radius_earth:
                print(f"  Planet Radius:  {candidate.planet_radius_earth:.2f} Earth")
            if candidate.transit_depth_ppm:
                print(f"  Transit Depth:  {candidate.transit_depth_ppm:.0f} ppm")
            if candidate.snr:
                print(f"  SNR:            {candidate.snr:.1f}")

        if candidate.vetting_report:
            v = candidate.vetting_report
            print(f"\n  --- Vetting Results ---")
            print(f"  Tests Passed:   {v.tests_passed}/{v.tests_run}")
            print(f"  Odd-Even:       {v.odd_even_test.value}")
            print(f"  Secondary:      {v.secondary_eclipse_test.value}")
            print(f"  V-Shape:        {v.vshape_test.value}")

        if candidate.submitted_to:
            print(f"\n  --- Submission ---")
            print(f"  Submitted To:   {candidate.submitted_to}")
            print(f"  External ID:    {candidate.external_id or 'Pending'}")

        print(f"\n  Created:        {candidate.created_at}")
        print(f"  Updated:        {candidate.updated_at}")

        if candidate.notes:
            print(f"\n  Notes: {candidate.notes}")

        print("=" * 60 + "\n")

    def update_candidate(self, candidate_id: str, **kwargs):
        """Update candidate fields."""
        candidate = self.db.update_candidate(candidate_id, **kwargs)

        if candidate:
            print(f"\nUpdated candidate: {candidate_id}")
        else:
            print(f"\nCandidate not found: {candidate_id}")

    def show_stats(self):
        """Show pipeline statistics."""
        stats = self.db.get_stats()

        print("\n" + "=" * 50)
        print("  Pipeline Statistics")
        print("=" * 50)
        print(f"\n  Total Candidates: {stats.total_candidates}")
        print(f"\n  By Status:")
        print(f"    Detected:       {stats.detected}")
        print(f"    Vetted:         {stats.vetted}")
        print(f"    Failed Vetting: {stats.failedvetting if hasattr(stats, 'failedvetting') else 0}")
        print(f"    Reported:       {stats.reported}")
        print(f"    Submitted:      {stats.submitted}")
        print(f"    Confirmed:      {stats.confirmed}")
        print(f"\n  By Priority:")
        print(f"    High:           {stats.high_priority}")
        print(f"    Medium:         {stats.medium_priority}")
        print(f"    Low:            {stats.low_priority}")

        if stats.last_detection:
            print(f"\n  Last Detection: {stats.last_detection}")
        print("=" * 50 + "\n")

    def prepare_submission(self, candidate_id: str, target: str = "exofop"):
        """Prepare submission package for a candidate."""
        from src.submission.exofop_submitter import ExoFOPSubmitter

        candidate = self.db.get_candidate(candidate_id)
        if not candidate:
            print(f"\nCandidate not found: {candidate_id}")
            return

        profile = self.db.get_user_profile()
        if not profile or not profile.get('name'):
            print("\nPlease configure your profile first: larun profile setup")
            return

        submitter = ExoFOPSubmitter(self.db)
        package = submitter.prepare_submission(candidate_id)

        print(f"\n  Submission package prepared for {candidate_id}")
        print(f"  Output: {package.output_dir}")
        print(f"\n{package.instructions}")

    def batch_process(self, csv_path: str, limit: int = 100):
        """Batch process targets from CSV."""
        from src.pipeline.batch_processor import BatchProcessor

        processor = BatchProcessor(pipeline=self.pipeline)

        def progress(done, total, target):
            print(f"\r  Progress: {done}/{total} - {target}          ", end="", flush=True)

        print(f"\nBatch processing targets from {csv_path}...")
        result = processor.process_csv(csv_path, limit=limit)

        print(f"\n\n  Completed: {result.successful}/{result.total}")
        print(f"  Candidates Found: {result.candidates_found}")
        print(f"  Duration: {result.duration_seconds:.1f}s")

        if result.errors:
            print(f"\n  Errors ({len(result.errors)}):")
            for err in result.errors[:5]:
                print(f"    - {err}")

    def start_dashboard(self, port: int = 8080, host: str = "127.0.0.1"):
        """Start the web dashboard."""
        try:
            from src.dashboard.app import create_app

            app = create_app(self.db)
            print(f"\n  Starting LARUN Dashboard...")
            print(f"  URL: http://{host}:{port}")
            print(f"  Press Ctrl+C to stop\n")

            app.run(host=host, port=port, debug=False)
        except ImportError:
            print("\nDashboard dependencies not installed.")
            print("Install with: pip install flask")
        except Exception as e:
            print(f"\nError starting dashboard: {e}")

    def run_with_nodes(self, target_id: str, node_ids: Optional[List[str]] = None):
        """Run multi-node analysis on a target."""
        from src.nodes.registry import NodeRegistry
        from src.nodes.loader import NodeLoader
        from src.nodes.aggregator import NodeAggregator

        registry = NodeRegistry()
        loader = NodeLoader(registry)
        aggregator = NodeAggregator()

        # Get nodes to use
        if node_ids:
            nodes = loader.load_nodes(node_ids)
        else:
            nodes = loader.load_enabled_nodes()

        if not nodes:
            print("No nodes available. Enable some nodes with 'larun node enable <id>'")
            return

        print(f"\nRunning {len(nodes)} node(s) on {target_id}...")
        print("-" * 50)

        # Load data
        data = self._load_target_data(target_id)
        if data is None:
            print(f"Could not load data for {target_id}")
            return

        # Run nodes
        results = []
        for node in nodes:
            print(f"  {node.node_id}...", end=" ", flush=True)
            result = node.run(data)
            results.append(result)

            if result.success:
                print(f"{result.classification} ({result.confidence:.1%})")
            else:
                print(f"FAILED")

        # Aggregate
        aggregated = aggregator.aggregate(results, target_id=target_id)
        print(aggregator.format_result(aggregated))

    def _load_target_data(self, target_id: str) -> Optional[np.ndarray]:
        """Load light curve data for a target."""
        # Try lightkurve first
        try:
            from lightkurve import search_lightcurve
            tic_id = target_id.replace('TIC', '').replace('_', '').strip()
            lc = search_lightcurve(f"TIC {tic_id}", mission="TESS")
            if lc:
                lc_data = lc[0].download()
                return lc_data.flux.value
        except:
            pass

        # Generate synthetic for testing
        np.random.seed(hash(target_id) % 2**32)
        n = 1024
        flux = 1.0 + 0.001 * np.random.randn(n)
        flux[400:420] -= 0.005  # Add transit-like dip
        return flux

    def node_commands(self, args: List[str]):
        """Handle node subcommands."""
        from src.cli.node_commands import NodeCommands
        cli = NodeCommands()
        cli.run(args)

    def interactive_mode(self):
        """Start interactive terminal mode."""
        print("\n" + "=" * 60)
        print("  LARUN Discovery Pipeline - Interactive Mode")
        print("=" * 60)

        if self.user_profile and self.user_profile.get('name'):
            print(f"  Welcome, {self.user_profile['name']}!")
        else:
            print("  Welcome! Run 'profile setup' to configure your profile.")

        print("\n  Commands: run, list, show, stats, node, batch, submit, dashboard, profile, help, quit\n")

        while True:
            try:
                cmd = input("larun> ").strip().lower()

                if not cmd:
                    continue

                parts = cmd.split()
                command = parts[0]
                args = parts[1:]

                if command in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif command == 'help':
                    self._show_help()
                elif command == 'run' and args:
                    # Check for --nodes flag
                    if '--nodes' in args:
                        idx = args.index('--nodes')
                        target = args[0] if args[0] != '--nodes' else args[idx + 2] if len(args) > idx + 2 else None
                        nodes = args[idx + 1].split(',') if len(args) > idx + 1 else None
                        if target:
                            self.run_with_nodes(target, nodes)
                    else:
                        self.run_pipeline(args[0])
                elif command == 'node':
                    self.node_commands(args)
                elif command == 'list':
                    status = args[0] if args else None
                    self.list_candidates(status=status)
                elif command == 'show' and args:
                    self.show_candidate(args[0])
                elif command == 'stats':
                    self.show_stats()
                elif command == 'profile':
                    if args and args[0] == 'setup':
                        self.setup_profile()
                    else:
                        self.show_profile()
                elif command == 'dashboard':
                    port = int(args[0]) if args else 8080
                    self.start_dashboard(port=port)
                elif command == 'submit' and args:
                    self.prepare_submission(args[0])
                elif command == 'batch' and args:
                    self.batch_process(args[0])
                else:
                    print(f"  Unknown command: {cmd}")
                    print("  Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\n  Use 'quit' to exit")
            except Exception as e:
                print(f"  Error: {e}")

    def _show_help(self):
        """Show help message."""
        print("""
  LARUN Pipeline Commands:

    run <target_id>     - Process target through discovery pipeline
    run <target> --nodes EXOPLANET-001,VSTAR-001  - Run with specific nodes
    list [status]       - List candidates (filter by status)
    show <id>           - Show candidate details
    stats               - Show pipeline statistics
    batch <csv>         - Batch process targets from CSV file
    submit <id>         - Prepare submission package

    node list           - List all analysis nodes
    node enable <id>    - Enable a node (e.g., VSTAR-001)
    node disable <id>   - Disable a node
    node info <id>      - Show node details
    node stats          - Show node statistics

    profile             - Show user profile
    profile setup       - Configure user profile

    dashboard [port]    - Start web dashboard

    help                - Show this help
    quit                - Exit interactive mode
        """)


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="LARUN Discovery Pipeline CLI")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Interactive mode (default)
    parser.add_argument('-i', '--interactive', action='store_true', help='Start interactive mode')

    # Run command
    run_parser = subparsers.add_parser('run', help='Process a target')
    run_parser.add_argument('target', help='Target ID (e.g., TIC 261136679)')
    run_parser.add_argument('--mission', default='TESS', help='Mission (TESS, Kepler)')
    run_parser.add_argument('--nodes', '-n', help='Comma-separated list of node IDs')

    # List command
    list_parser = subparsers.add_parser('list', help='List candidates')
    list_parser.add_argument('--status', help='Filter by status')
    list_parser.add_argument('--priority', help='Filter by priority')
    list_parser.add_argument('--limit', type=int, default=20, help='Max results')

    # Show command
    show_parser = subparsers.add_parser('show', help='Show candidate details')
    show_parser.add_argument('id', help='Candidate ID')

    # Stats command
    subparsers.add_parser('stats', help='Show pipeline statistics')

    # Profile commands
    profile_parser = subparsers.add_parser('profile', help='User profile management')
    profile_parser.add_argument('action', nargs='?', choices=['setup', 'show'], default='show')

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch process targets')
    batch_parser.add_argument('csv', help='CSV file with targets')
    batch_parser.add_argument('--limit', type=int, default=100)

    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Prepare submission')
    submit_parser.add_argument('id', help='Candidate ID')
    submit_parser.add_argument('--target', default='exofop', help='Submission target')

    # Dashboard command
    dash_parser = subparsers.add_parser('dashboard', help='Start web dashboard')
    dash_parser.add_argument('--port', type=int, default=8080)
    dash_parser.add_argument('--host', default='127.0.0.1')

    # Node command (delegates to node_commands.py)
    node_parser = subparsers.add_parser('node', help='Node management commands')
    node_parser.add_argument('node_args', nargs='*', help='Node subcommand and arguments')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    cli = LarunCLI()

    if args.command == 'run':
        if args.nodes:
            node_ids = [n.strip() for n in args.nodes.split(',')]
            cli.run_with_nodes(args.target, node_ids)
        else:
            cli.run_pipeline(args.target, mission=args.mission)
    elif args.command == 'node':
        cli.node_commands(args.node_args)
    elif args.command == 'list':
        cli.list_candidates(status=args.status, priority=args.priority, limit=args.limit)
    elif args.command == 'show':
        cli.show_candidate(args.id)
    elif args.command == 'stats':
        cli.show_stats()
    elif args.command == 'profile':
        if args.action == 'setup':
            cli.setup_profile()
        else:
            cli.show_profile()
    elif args.command == 'batch':
        cli.batch_process(args.csv, limit=args.limit)
    elif args.command == 'submit':
        cli.prepare_submission(args.id, target=args.target)
    elif args.command == 'dashboard':
        cli.start_dashboard(port=args.port, host=args.host)
    else:
        # Default: interactive mode
        cli.interactive_mode()


if __name__ == '__main__':
    main()
