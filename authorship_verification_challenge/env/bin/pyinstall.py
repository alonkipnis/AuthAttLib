#!/Users/kipnisal/python-virtual-environments/env/bin/python3
import sys
import os
import optparse
import pkg_resources
import urllib2
import urllib
import mimetypes
import zipfile
import tarfile
import tempfile
import subprocess
import posixpath
import re
import shutil
try:
    from hashlib import md5
except ImportError:
    import md5 as md5_module
    md5 = md5_module.new
import urlparse
from email.FeedParser import FeedParser
import traceback
from cStringIO import StringIO
import socket
from Queue import Queue
from Queue import Empty as QueueEmpty
import threading
import httplib
import time
import logging

class InstallationError(Exception):
    """General exception during installation"""

class DistributionNotFound(InstallationError):
    """Raised when a distribution cannot be found to satisfy a requirement"""

if getattr(sys, 'real_prefix', None):
    ## FIXME: is build/ a good name?
    base_prefix = os.path.join(sys.prefix, 'build')
    base_src_prefix = os.path.join(sys.prefix, 'src')
else:
    ## FIXME: this isn't a very good default
    base_prefix = os.path.join(os.getcwd(), 'build')
    base_src_prefix = os.path.join(os.getcwd(), 'src')

pypi_url = "http://pypi.python.org/simple"

default_timeout = 15

parser = optparse.OptionParser(
    usage='%prog [OPTIONS] PACKAGE_NAMES')

parser.add_option(
    '-e', '--editable',
    dest='editables',
    action='append',
    default=[],
    metavar='svn+REPOS_URL[@REV]#egg=PACKAGE',
    help='Install a package directly from a checkout.  Source will be checked '
    'out into src/PACKAGE (lower-case) and installed in-place (using '
    'setup.py develop).  This option may be provided multiple times.')
parser.add_option(
    '-r', '--requirement',
    dest='requirements',
    action='append',
    default=[],
    metavar='FILENAME',
    help='Install all the packages listed in the given requirements file.  '
    'This option can be used multiple times.')

parser.add_option(
    '-f', '--find-links',
    dest='find_links',
    action='append',
    default=[],
    metavar='URL',
    help='URL to look for packages at')
parser.add_option(
    '-i', '--index-url',
    dest='index_url',
    metavar='URL',
    default=pypi_url,
    help='base URL of Python Package Index')
parser.add_option(
    '--extra-index-url',
    dest='extra_index_urls',
    metavar='URL',
    action='append',
    default=[],
    help='extra URLs of package indexes to use in addition to --index-url')

parser.add_option(
    '-b', '--build', '--build-dir', '--build-directory',
    dest='build_dir',
    metavar='DIR',
    default=None,
    help='Unpack packages into DIR (default %s) and build from there' % base_prefix)
parser.add_option(
    '--src', '--source',
    dest='src_dir',
    metavar='DIR',
    default=None,
    help='Check out --editable packages into DIR (default %s)' % base_src_prefix)
parser.add_option(
    '--timeout',
    metavar='SECONDS',
    dest='timeout',
    type='float',
    default=default_timeout,
    help='Set the socket timeout (default %s seconds)' % default_timeout)

parser.add_option(
    '-U', '--upgrade',
    dest='upgrade',
    action='store_true',
    help='Upgrade all packages to the newest available version')
parser.add_option(
    '-I', '--ignore-installed',
    dest='ignore_installed',
    action='store_true',
    help='Ignore the installed packages (reinstalling instead)')
parser.add_option(
    '--no-install',
    dest='no_install',
    action='store_true',
    help="Download and unpack all packages, but don't actually install them")

parser.add_option(
    '--bundle',
    dest='bundle',
    metavar='BUNDLE_FILE',
    help="Collect all packages and create a .pybundle file.")
parser.add_option(
    '--freeze',
    dest='freeze',
    metavar='FREEZE_FILE',
    help="Create a file that can be used with --requirement to reproduce the "
    "installed packages.  You can also give one --requirement file that will "
    "be used as the basis of the new file.")

parser.add_option(
    '-E', '--environment',
    dest='venv',
    metavar='DIR',
    help='virtualenv environment to run pyinstall in (either give the '
    'interpreter or the environment base directory)')

parser.add_option(
    '-v', '--verbose',
    dest='verbose',
    action='count',
    default=0,
    help='Give more output')
parser.add_option(
    '-q', '--quiet',
    dest='quiet',
    action='count',
    default=0,
    help='Give less output')
parser.add_option(
    '--log',
    dest='log',
    metavar='FILENAME',
    help='Log file where a complete (maximum verbosity) record will be kept')

parser.add_option(
    '--proxy',
    dest='proxy',
    type='str',
    default='',
    help="Specify a proxy in the form user:passwd@proxy.server:port. "
    "Note that the user:password@ is optional and required only if you "
    "are behind an authenticated proxy.  If you provide "
    "user@proxy.server:port then you will be prompted for a password."
    )

parser.add_option(
    '--install-option',
    dest='install_options',
    action='append',
    help="Extra arguments to be supplied to the setup.py install "
    "command (use like --install-option=\"--install-scripts=/usr/local/bin\").  "
    "Use multiple --install-option options to pass multiple options to setup.py install"
    )

def get_proxy(proxystr=''):
    """Get the proxy given the option passed on the command line.  If an
    empty string is passed it looks at the HTTP_PROXY environment
    variable."""
    if not proxystr:
        proxystr = os.environ.get('HTTP_PROXY', '')
    if proxystr:
        if '@' in proxystr:
            user_password, server_port = proxystr.split('@', 1)
            if ':' in user_password:
                user, password = user_password.split(':', 1)
            else:
                user = user_password
                import getpass
                prompt = 'Password for %s@%s: ' % (user, server_port)
                password = urllib.quote(getpass.getpass(prompt))
            return '%s:%s@%s' % (user, password, server_port)
        else:
            return proxystr
    else:
        return None

def setup_proxy_handler(proxystr=''):
    """Set the proxy handler given the option passed on the command
    line.  If an empty string is passed it looks at the HTTP_PROXY
    environment variable.  """
    proxy = get_proxy(proxystr)
    if proxy:
        proxy_support = urllib2.ProxyHandler({"http": proxy, "ftp": proxy})
        opener = urllib2.build_opener(proxy_support, urllib2.CacheFTPHandler)
        urllib2.install_opener(opener)


def main(initial_args=None):
    global logger
    if initial_args is None:
        initial_args = sys.argv[1:]
    options, args = parser.parse_args(initial_args)

    if args and args[-1] == '___VENV_RESTART___':
        ## FIXME: We don't do anything this this value yet:
        venv_location = args[-2]
        args = args[:-2]
        options.venv = None
    level = 1 # Notify
    level += options.verbose
    level -= options.quiet
    level = Logger.level_for_integer(4-level)
    complete_log = []
    logger = Logger([(level, sys.stdout), 
                     (Logger.DEBUG, complete_log.append)])
    if options.venv:
        if options.verbose > 0:
            # The logger isn't setup yet
            print 'Running in environment %s' % options.venv
        restart_in_venv(options.venv, initial_args)
        # restart_in_venv should actually never return, but for clarity...
        return
    ## FIXME: not sure if this sure come before or after venv restart
    if options.log:
        log_fp = open_logfile_append(options.log)
        logger.consumers.append((logger.DEBUG, log_fp))
    else:
        log_fp = None

    socket.setdefaulttimeout(options.timeout or None)

    setup_proxy_handler(options.proxy)

    if options.bundle:
        if not options.build_dir:
            options.build_dir = backup_dir(base_prefix, '-bundle')
        if not options.src_dir:
            options.src_dir = backup_dir(base_src_prefix, '-bundle')
        # We have to get everything when creating a bundle:
        options.ignore_installed = True
        logger.notify('Putting temporary build files in %s and source/develop files in %s'
                      % (display_path(options.build_dir), display_path(options.src_dir)))
    if not options.build_dir:
        options.build_dir = base_prefix
    if not options.src_dir:
        options.src_dir = base_src_prefix
    options.build_dir = os.path.abspath(options.build_dir)
    options.src_dir = os.path.abspath(options.src_dir)
    install_options = options.install_options or []
    try:
        if options.freeze:
            if options.requirements:
                if len(options.requirements) > 1:
                    raise InstallationError(
                        "When using --freeze you can only provide one --requirement option")
                requirement = options.requirements[0]
            else:
                requirement = None
            write_freeze(
                options.freeze,
                requirement=requirement,
                find_links=options.find_links)
            return
        index_urls = [options.index_url] + options.extra_index_urls
        finder = PackageFinder(
            find_links=options.find_links,
            index_urls=index_urls)
        requirement_set = RequirementSet(build_dir=options.build_dir,
                                         src_dir=options.src_dir,
                                         upgrade=options.upgrade,
                                         ignore_installed=options.ignore_installed)
        for name in args:
            requirement_set.add_requirement(
                InstallRequirement.from_line(name, None))
        for name in options.editables:
            requirement_set.add_requirement(
                InstallRequirement.from_editable(name))
        for filename in options.requirements:
            for req in parse_requirements(filename, finder=finder):
                requirement_set.add_requirement(req)
        exit = 0
        requirement_set.install_files(finder)
        if not options.no_install and not options.bundle:
            requirement_set.install(install_options)
            logger.notify('Successfully installed %s' % requirement_set)
        elif options.bundle:
            requirement_set.create_bundle(options.bundle)
            logger.notify('Created bundle in %s' % options.bundle)
        else:
            logger.notify('Successfully downloaded %s' % requirement_set)
    except InstallationError, e:
        logger.fatal(str(e))
        logger.info('Exception information:\n%s' % format_exc())
        exit = 1
    except:
        logger.fatal('Exception:\n%s' % format_exc())
        exit = 2
    if log_fp is not None:
        log_fp.close()
    if exit:
        log_fn = './pyinstall-log.txt'
        text = '\n'.join(complete_log)
        logger.fatal('Storing complete log in %s' % log_fn)
        log_fp = open_logfile_append(log_fn)
        log_fp.write(text)
        log_fp.close()
    sys.exit(exit)

def format_exc(exc_info=None):
    if exc_info is None:
        exc_info = sys.exc_info()
    out = StringIO()
    traceback.print_exception(*exc_info, **dict(file=out))
    return out.getvalue()

def restart_in_venv(venv, args):
    """
    Restart this script using the interpreter in the given virtual environment
    """
    venv = os.path.abspath(venv)
    if not os.path.exists(venv):
        try:
            import virtualenv
        except ImportError:
            print 'The virtual environment does not exist: %s' % venv
            print 'and virtualenv is not installed, so a new environment cannot be created'
            sys.exit(3)
        print 'Creating new virtualenv environment in %s' % venv
        virtualenv.logger = logger
        logger.indent += 2
        ## FIXME: always have no_site_packages?
        virtualenv.create_environment(venv, site_packages=False)
    if sys.platform == 'win32':
        python = os.path.join(venv, 'Scripts', 'python')
    else:
        python = os.path.join(venv, 'bin', 'python')
    if not os.path.exists(python):
        python = venv
    if not os.path.exists(python):
        raise BadCommand('Cannot find virtual environment interpreter at %s' % python)
    base = os.path.dirname(os.path.dirname(python))
    os.execv(python, [python, __file__] + args + [base, '___VENV_RESTART___'])

class PackageFinder(object):
    """This finds packages.

    This is meant to match easy_install's technique for looking for
    packages, by reading pages and looking for appropriate links
    """

    failure_limit = 3

    def __init__(self, find_links, index_urls):
        self.find_links = find_links
        self.index_urls = index_urls
        self.dependency_links = []
        self.cache = PageCache()
    
    def add_dependency_links(self, links):
        ## FIXME: this shouldn't be global list this, it should only
        ## apply to requirements of the package that specifies the
        ## dependency_links value
        ## FIXME: also, we should track comes_from (i.e., use Link)
        self.dependency_links.extend(links)

    def find_requirement(self, req, upgrade):
        url_name = req.url_name
        # Check that we have the url_name correctly spelled:
        main_index_url = Link(posixpath.join(self.index_urls[0], url_name))
        # This will also cache the page, so it's okay that we get it again later:
        page = self._get_page(main_index_url, req)
        if page is None:
            url_name = self._find_url_name(Link(self.index_urls[0]), url_name, req)
        if url_name is not None:
            locations = [
                posixpath.join(url, url_name)
                for url in self.index_urls] + self.find_links
        else:
            locations = list(self.find_links)
        locations.extend(self.dependency_links)
        for version in req.absolute_versions:
            locations = [
                posixpath.join(url, url_name, version)] + locations
        locations = [Link(url) for url in locations]
        logger.debug('URLs to search for versions for %s:' % req)
        for location in locations:
            logger.debug('* %s' % location)
        found_versions = []
        for page in self._get_pages(locations, req):
            logger.debug('Analyzing links from page %s' % page.url)
            logger.indent += 2
            try:
                found_versions.extend(self._package_versions(page.links, req.name.lower()))
            finally:
                logger.indent -= 2
        dependency_versions = list(self._package_versions([Link(url) for url in self.dependency_links], req.name.lower()))
        if dependency_versions:
            logger.info('dependency_links found: %s' % ', '.join([link.url for parsed, link, version in dependency_versions]))
            found_versions.extend(dependency_versions)
        if not found_versions:
            logger.fatal('Could not find any downloads that satisfy the requirement %s' % req)
            raise DistributionNotFound('No distributions at all found for %s' % req)
        if req.satisfied_by is not None:
            found_versions.append((req.satisfied_by.parsed_version, Inf, req.satisfied_by.version))
        found_versions.sort(reverse=True)
        applicable_versions = []
        for (parsed_version, link, version) in found_versions:
            if version not in req.req:
                logger.info("Ignoring link %s, version %s doesn't match %s"
                            % (link, version, ','.join([''.join(s) for s in req.req.specs])))
                continue
            applicable_versions.append((link, version))
        existing_applicable = bool([link for link, version in applicable_versions if link is Inf])
        if not upgrade and existing_applicable:
            if applicable_versions[0][1] is Inf:
                logger.info('Existing installed version (%s) is most up-to-date and satisfies requirement'
                            % req.satisfied_by.version)
            else:
                logger.info('Existing installed version (%s) satisfies requirement (most up-to-date version is %s)'
                            % (req.satisfied_by.version, application_versions[0][2]))
            return None
        if not applicable_versions:
            logger.fatal('Could not find a version that satisfies the requirement %s (from versions: %s)'
                         % (req, ', '.join([version for parsed_version, link, version in found_versions])))
            raise DistributionNotFound('No distributions matching the version for %s' % req)
        if applicable_versions[0][0] is Inf:
            # We have an existing version, and its the best version
            logger.info('Installed version (%s) is most up-to-date (past versions: %s)'
                        % (req.satisfied_by.version, ', '.join([version for link, version in applicable_versions[1:]]) or 'none'))
            return None
        if len(applicable_versions) > 1:
            logger.info('Using version %s (newest of versions: %s)' %
                        (applicable_versions[0][1], ', '.join([version for link, version in applicable_versions])))
        return applicable_versions[0][0]

    def _find_url_name(self, index_url, url_name, req):
        """Finds the true URL name of a package, when the given name isn't quite correct.
        This is usually used to implement case-insensitivity."""
        if not index_url.url.endswith('/'):
            # Vaguely part of the PyPI API... weird but true.
            ## FIXME: bad to modify this?
            index_url.url += '/'
        page = self._get_page(index_url, req)
        if page is None:
            logger.fatal('Cannot fetch index base URL %s' % index_url)
            raise DistributionNotFound('Cannot find requirement %s, nor fetch index URL %s' % (req, index_url))
        norm_name = normalize_name(req.url_name)
        for link in page.links:
            base = posixpath.basename(link.path.rstrip('/'))
            if norm_name == normalize_name(base):
                logger.notify('Real name of requirement %s is %s' % (url_name, base))
                return base
        return None

    def _get_pages(self, locations, req):
        """Yields (page, page_url) from the given locations, skipping
        locations that have errors, and adding download/homepage links"""
        pending_queue = Queue()
        for location in locations:
            pending_queue.put(location)
        done = []
        seen = set()
        threads = []
        for i in range(min(10, len(locations))):
            t = threading.Thread(target=self._get_queued_page, args=(req, pending_queue, done, seen))
            t.setDaemon(True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        return done

    _log_lock = threading.Lock()

    def _get_queued_page(self, req, pending_queue, done, seen):
        while 1:
            try:
                location = pending_queue.get(False)
            except QueueEmpty:
                return
            if location in seen:
                continue
            seen.add(location)
            page = self._get_page(location, req)
            if page is None:
                continue
            done.append(page)
            for link in page.rel_links():
                pending_queue.put(link)

    _egg_fragment_re = re.compile(r'#egg=([^&]*)')
    _egg_info_re = re.compile(r'([a-z0-9_.]+)-([a-z0-9_.-]+)', re.I)
    _py_version_re = re.compile(r'-py([123]\.[0-9])$')

    def _package_versions(self, links, search_name):
        seen_links = {}
        for link in links:
            if link.url in seen_links:
                continue
            seen_links[link.url] = None
            if link.egg_fragment:
                egg_info = link.egg_fragment
            else:
                path = link.path
                egg_info, ext = link.splitext()
                if not ext:
                    logger.debug('Skipping link %s; not a file' % link)
                    continue
                if egg_info.endswith('.tar'):
                    # Special double-extension case:
                    egg_info = egg_info[:-4]
                    ext = '.tar' + ext
                if ext not in ('.tar.gz', '.tar.bz2', '.tar', '.tgz', '.zip'):
                    logger.debug('Skipping link %s; unknown archive format: %s' % (link, ext))
                    continue
            version = self._egg_info_matches(egg_info, search_name, link)
            if version is None:
                logger.debug('Skipping link %s; wrong project name (not %s)' % (link, search_name))
                continue
            match = self._py_version_re.search(version)
            if match:
                version = version[:match.start()]
                py_version = match.group(1)
                if py_version != sys.version[:3]:
                    logger.debug('Skipping %s because Python version is incorrect' % link)
                    continue
            logger.debug('Found link %s, version: %s' % (link, version))
            yield (pkg_resources.parse_version(version),
                   link,
                   version)

    def _egg_info_matches(self, egg_info, search_name, link):
        match = self._egg_info_re.search(egg_info)
        if not match:
            logger.debug('Could not parse version from link: %s' % link)
            return None
        name = match.group(0).lower()
        # To match the "safe" name that pkg_resources creates:
        name = name.replace('_', '-')
        if name.startswith(search_name.lower()):
            return match.group(0)[len(search_name):].lstrip('-')
        else:
            return None

    def _get_page(self, link, req):
        return HTMLPage.get_page(link, req, cache=self.cache)


class InstallRequirement(object):

    def __init__(self, req, comes_from, source_dir=None, editable=False,
                 url=None, update=True):
        if isinstance(req, basestring):
            req = pkg_resources.Requirement.parse(req)
        self.req = req
        self.comes_from = comes_from
        self.source_dir = source_dir
        self.editable = editable
        if editable:
            assert url, "You must give url with editable=True"
        self.url = url
        self._egg_info_path = None
        # This holds the pkg_resources.Distribution object if this requirement
        # is already available:
        self.satisfied_by = None
        self._temp_build_dir = None
        self._is_bundle = None
        # True if the editable should be updated:
        self.update = update

    @classmethod
    def from_editable(cls, editable_req, comes_from=None):
        name, url = parse_editable(editable_req)
        return cls(name, comes_from, editable=True, url=url)

    @classmethod
    def from_line(cls, name, comes_from=None):
        """Creates an InstallRequirement from a name, which might be a
        requirement, filename, or URL.
        """
        url = None
        req = name
        if is_url(name):
            url = name
            ## FIXME: I think getting the requirement here is a bad idea:
            #req = get_requirement_from_url(url)
            req = None
        elif is_filename(name):
            if not os.path.exists(name):
                logger.warn('Requirement %r looks like a filename, but the file does not exist'
                            % name)
            url = filename_to_url(name)
            #req = get_requirement_from_url(url)
            req = None
        return cls(req, comes_from, url=url)

    def __str__(self):
        if self.req:
            s = str(self.req)
            if self.url:
                s += ' from %s' % self.url
        else:
            s = self.url
        if self.satisfied_by is not None:
            s += ' in %s' % display_path(self.satisfied_by.location)
        if self.editable:
            if self.req:
                s += ' checkout from %s' % self.url
        if self.comes_from:
            if isinstance(self.comes_from, basestring):
                comes_from = self.comes_from
            else:
                comes_from = self.comes_from.from_path()
            s += ' (from %s)' % comes_from
        return s

    def from_path(self):
        s = str(self.req)
        if self.comes_from:
            if isinstance(self.comes_from, basestring):
                comes_from = self.comes_from
            else:
                comes_from = self.comes_from.from_path()
            s += '->' + comes_from
        return s

    def build_location(self, build_dir):
        if self._temp_build_dir is not None:
            return self._temp_build_dir
        if self.req is None:
            self._temp_build_dir = tempfile.mkdtemp('-build', 'pyinstall-')
            return self._temp_build_dir
        if self.editable:
            name = self.name.lower()
        else:
            name = self.name
        return os.path.join(build_dir, name)

    @property
    def name(self):
        if self.req is None:
            return None
        return self.req.project_name

    @property
    def url_name(self):
        if self.req is None:
            return None
        return urllib.quote(self.req.unsafe_name)

    @property
    def setup_py(self):
        return os.path.join(self.source_dir, 'setup.py')

    def run_egg_info(self):
        assert self.source_dir
        if self.name:
            logger.notify('Running setup.py egg_info for package %s' % self.name)
        else:
            logger.notify('Running setup.py egg_info for package from %s' % self.url)
        logger.indent += 2
        try:
            script = self._run_setup_py
            script = script.replace('__SETUP_PY__', repr(self.setup_py))
            script = script.replace('__PKG_NAME__', repr(self.name))
            # We can't put the .egg-info files at the root, because then the source code will be mistaken
            # for an installed egg, causing problems
            if self.editable:
                egg_base_option = []
            else:
                egg_info_dir = os.path.join(self.source_dir, 'pyinstall-egg-info')
                if not os.path.exists(egg_info_dir):
                    os.makedirs(egg_info_dir)
                egg_base_option = ['--egg-base', 'pyinstall-egg-info']
            call_subprocess(
                [sys.executable, '-c', script, 'egg_info'] + egg_base_option,
                cwd=self.source_dir, filter_stdout=self._filter_install, show_stdout=False,
                command_level=Logger.VERBOSE_DEBUG,
                command_desc='python setup.py egg_info')
        finally:
            logger.indent -= 2
        if not self.req:
            self.req = pkg_resources.Requirement.parse(self.pkg_info()['Name'])

    ## FIXME: This is a lame hack, entirely for PasteScript which has
    ## a self-provided entry point that causes this awkwardness
    _run_setup_py = """
__file__ = __SETUP_PY__
from setuptools.command import egg_info
def replacement_run(self):
    self.mkpath(self.egg_info)
    installer = self.distribution.fetch_build_egg
    for ep in egg_info.iter_entry_points('egg_info.writers'):
        # require=False is the change we're making:
        writer = ep.load(require=False)
        writer(self, ep.name, egg_info.os.path.join(self.egg_info,ep.name))
    self.find_sources()
egg_info.egg_info.run = replacement_run
execfile(__file__)
"""

    def egg_info_data(self, filename):
        if self.satisfied_by is not None:
            if not self.satisfied_by.has_metadata(filename):
                return None
            return self.satisfied_by.get_metadata(filename)
        assert self.source_dir
        filename = self.egg_info_path(filename)
        if not os.path.exists(filename):
            return None
        fp = open(filename, 'r')
        data = fp.read()
        fp.close()
        return data

    def egg_info_path(self, filename):
        if self._egg_info_path is None:
            if self.editable:
                base = self.source_dir
            else:
                base = os.path.join(self.source_dir, 'pyinstall-egg-info')
            filenames = os.listdir(base)
            if self.editable:
                filenames = [f for f in filenames if f.endswith('.egg-info')]
            assert len(filenames) == 1, "Unexpected files/directories in %s: %s" % (base, ' '.join(filenames))
            self._egg_info_path = os.path.join(base, filenames[0])
        return os.path.join(self._egg_info_path, filename)

    def egg_info_lines(self, filename):
        data = self.egg_info_data(filename)
        if not data:
            return []
        result = []
        for line in data.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            result.append(line)
        return result

    def pkg_info(self):
        p = FeedParser()
        data = self.egg_info_data('PKG-INFO')
        if not data:
            logger.warn('No PKG-INFO file found in %s' % display_path(self.egg_info_path('PKG-INFO')))
        p.feed(data or '')
        return p.close()

    @property
    def dependency_links(self):
        return self.egg_info_lines('dependency_links.txt')

    _requirements_section_re = re.compile(r'\[(.*?)\]')

    def requirements(self, extras=()):
        in_extra = None
        for line in self.egg_info_lines('requires.txt'):
            match = self._requirements_section_re.match(line)
            if match:
                in_extra = match.group(1)
                continue
            if in_extra and in_extra not in extras:
                # Skip requirement for an extra we aren't requiring
                continue
            yield line

    @property
    def absolute_versions(self):
        for qualifier, version in self.req.specs:
            if qualifier == '==':
                yield version

    @property
    def installed_version(self):
        return self.pkg_info()['version']

    def assert_source_matches_version(self):
        assert self.source_dir
        if self.comes_from == 'command line':
            # We don't check the versions of things explicitly installed.
            # This makes, e.g., "pyinstall Package==dev" possible
            return
        version = self.installed_version
        if version not in self.req:
            logger.fatal(
                'Source in %s has the version %s, which does not match the requirement %s'
                % (display_path(self.source_dir), version, self))
            raise InstallationError(
                'Source in %s has version %s that conflicts with %s' 
                % (display_path(self.source_dir), version, self))
        else:
            logger.debug('Source in %s has version %s, which satisfies requirement %s'
                         % (display_path(self.source_dir), version, self))

    def update_editable(self):
        assert self.editable and self.url
        assert self.source_dir
        assert '+' in self.url, "bad url: %r" % self.url
        if not self.update:
            return
        vc_type, url = self.url.split('+', 1)
        vc_type = vc_type.lower()
        if vc_type == 'svn':
            self.checkout_svn()
        else:
            assert 0, (
                'Unexpected version control type (in %s): %s' 
                % (self.url, vc_type))

    def checkout_svn(self):
        url = self.url.split('+', 1)[1]
        url = url.split('#', 1)[0]
        if '@' in url:
            url, rev = url.split('@', 1)
        else:
            rev = None
        if rev:
            rev_options = ['-r', rev]
            rev_display = ' (to revision %s)' % rev
        else:
            rev_options = []
            rev_display = ''
        dest = self.source_dir
        checkout = True
        if os.path.exists(os.path.join(self.source_dir, '.svn')):
            existing_url = _get_svn_info(self.source_dir)[0]
            checkout = False
            if existing_url == url:
                logger.info('Checkout in %s exists, and has correct URL (%s)'
                            % (display_path(self.source_dir), url))
                logger.notify('Updating checkout %s%s' % (display_path(self.source_dir), rev_display))
                call_subprocess(
                    ['svn', 'update'] + rev_options + [self.source_dir])
            else:
                logger.warn('svn checkout in %s exists with URL %s' % (display_path(self.source_dir), existing_url))
                logger.warn('The plan is to install the svn repository %s' % url)
                response = ask('What to do?  (s)witch, (i)gnore, (w)ipe, (b)ackup', ('s', 'i', 'w', 'b'))
                if response == 's':
                    logger.notify('Switching checkout %s to %s%s'
                                  % (display_path(self.source_dir), url, rev_display))
                    call_subprocess(
                        ['svn', 'switch'] + rev_options + [url, self.source_dir])
                elif response == 'i':
                    # do nothing
                    pass
                elif response == 'w':
                    logger.warn('Deleting %s' % display_path(self.source_dir))
                    shutil.rmtree(self.source_dir)
                    checkout = True
                elif response == 'b':
                    dest_dir = backup_dir(self.source_dir)
                    logger.warn('Backing up %s to %s' % display_path(self.source_dir, dest_dir))
                    shutil.move(self.source_dir, dest_dir)
                    checkout = True
        if checkout:
            logger.notify('Checking out %s%s to %s' % (url, rev_display, display_path(self.source_dir)))
            call_subprocess(
                ['svn', 'checkout', '-q'] + rev_options + [url, self.source_dir])

    def install(self, install_options):
        if self.editable:
            self.install_editable()
            return
        ## FIXME: this is not a useful record:
        ## Also a bad location
        ## And not right on Windows
        install_location = os.path.join(sys.prefix, 'lib', 'python%s' % sys.version[:3])
        record_filename = os.path.join(install_location, 'install-record-%s.txt' % self.name)
        ## FIXME: I'm not sure if this is a reasonable location; probably not
        ## but we can't put it in the default location, as that is a virtualenv symlink that isn't writable
        header_dir = os.path.join(os.path.dirname(os.path.dirname(self.source_dir)), 'lib', 'include')
        logger.notify('Running setup.py install for %s' % self.name)
        logger.indent += 2
        try:
            call_subprocess(
                [sys.executable, '-c',
                 "import setuptools; __file__=%r; execfile(%r)" % (self.setup_py, self.setup_py), 
                 'install', '--single-version-externally-managed', '--record', record_filename,
                 '--install-headers', header_dir] + install_options,
                cwd=self.source_dir, filter_stdout=self._filter_install, show_stdout=False)
        finally:
            logger.indent -= 2

    def remove_temporary_source(self):
        """Remove the source files from this requirement, if they are marked
        for deletion"""
        if self.is_bundle or os.path.exists(self.delete_marker_filename):
            logger.info('Removing source in %s' % self.source_dir)
            if self.source_dir:
                shutil.rmtree(self.source_dir)
            self.source_dir = None
            if self._temp_build_dir and os.path.exists(self._temp_build_dir):
                shutil.rmtree(self._temp_build_dir)
            self._temp_build_dir = None

    def install_editable(self):
        logger.notify('Running setup.py develop for %s' % self.name)
        logger.indent += 2
        try:
            ## FIXME: should we do --install-headers here too?
            call_subprocess(
                [sys.executable, '-c',
                 "import setuptools; __file__=%r; execfile(%r)" % (self.setup_py, self.setup_py),
                 'develop', '--no-deps'], cwd=self.source_dir, filter_stdout=self._filter_install,
                show_stdout=False)
        finally:
            logger.indent -= 2

    def _filter_install(self, line):
        level = Logger.NOTIFY
        for regex in [r'^running .*', r'^writing .*', '^creating .*', '^[Cc]opying .*',
                      r'^reading .*', r"^removing .*\.egg-info' \(and everything under it\)$",
                      r'^byte-compiling ',
                      # Not sure what this warning is, but it seems harmless:
                      r"^warning: manifest_maker: standard file '-c' not found$"]:
            if re.search(regex, line.strip()):
                level = Logger.INFO
                break
        return (level, line)

    def check_if_exists(self):
        """Checks if this requirement is satisfied by something already installed"""
        if self.req is None:
            return False
        try:
            dist = pkg_resources.get_distribution(self.req)
        except pkg_resources.DistributionNotFound:
            return False
        self.satisfied_by = dist
        return True

    @property
    def is_bundle(self):
        if self._is_bundle is not None:
            return self._is_bundle
        base = self._temp_build_dir
        if not base:
            ## FIXME: this doesn't seem right:
            return False
        self._is_bundle = os.path.exists(os.path.join(base, 'pyinstall-manifest.txt'))
        return self._is_bundle
    
    def bundle_requirements(self):
        base = self._temp_build_dir
        assert base
        src_dir = os.path.join(base, 'src')
        build_dir = os.path.join(base, 'build')
        if os.path.exists(src_dir):
            for package in os.listdir(src_dir):
                ## FIXME: svnism:
                url = 'svn+' + _get_svn_info(os.path.join(src_dir, package))[0]
                yield InstallRequirement(
                    package, self, editable=True, url=url,
                    update=False)
        if os.path.exists(build_dir):
            for package in os.listdir(build_dir):
                yield InstallRequirement(
                    package, self)

    def move_bundle_files(self, dest_build_dir, dest_src_dir):
        base = self._temp_build_dir
        assert base
        src_dir = os.path.join(base, 'src')
        build_dir = os.path.join(base, 'build')
        for source_dir, dest_dir in [(src_dir, dest_src_dir),
                                     (build_dir, dest_build_dir)]:
            if os.path.exists(source_dir):
                for dirname in os.listdir(source_dir):
                    dest = os.path.join(dest_dir, dirname)
                    if os.path.exists(dest):
                        logger.warn('The directory %s (containing package %s) already exists; cannot move source from bundle %s'
                                    % (dest, dirname, self))
                        continue
                    if not os.path.exists(dest_dir):
                        logger.info('Creating directory %s' % dest_dir)
                        os.makedirs(dest_dir)
                    shutil.move(os.path.join(source_dir, dirname), dest)

    @property
    def delete_marker_filename(self):
        assert self.source_dir
        return os.path.join(self.source_dir, 'pyinstall-delete-this-directory.txt')

DELETE_MARKER_MESSAGE = '''\
This file is placed here by pyinstall to indicate the source was put
here by pyinstall.

Once this package is successfully installed this source code will be
deleted (unless you remove this file).
'''

class RequirementSet(object):

    def __init__(self, build_dir, src_dir, upgrade=False, ignore_installed=False):
        self.build_dir = build_dir
        self.src_dir = src_dir
        self.upgrade = upgrade
        self.ignore_installed = ignore_installed
        self.requirements = {}
        # Mapping of alias: real_name
        self.requirement_aliases = {}
        self.unnamed_requirements = []

    def __str__(self):
        reqs = [req for req in self.requirements.values()
                if not req.comes_from]
        reqs.sort(key=lambda req: req.name.lower())
        return ' '.join([str(req.req) for req in reqs])

    def add_requirement(self, install_req):
        name = install_req.name
        if not name:
            self.unnamed_requirements.append(install_req)
        else:
            if self.has_requirement(name):
                raise InstallationError(
                    'Double requirement given: %s (aready in %s, name=%r)'
                    % (install_req, self.get_requirement(name), name))
            self.requirements[name] = install_req
            ## FIXME: what about other normalizations?  E.g., _ vs. -?
            if name.lower() != name:
                self.requirement_aliases[name.lower()] = name

    def has_requirement(self, project_name):
        for name in project_name, project_name.lower():
            if name in self.requirements or name in self.requirement_aliases:
                return True
        return False

    def get_requirement(self, project_name):
        for name in project_name, project_name.lower():
            if name in self.requirements:
                return self.requirements[name]
            if name in self.requirement_aliases:
                return self.requirements[self.requirement_aliases[name]]
        raise KeyError("No project with the name %r" % project_name)

    def install_files(self, finder):
        unnamed = list(self.unnamed_requirements)
        reqs = self.requirements.values()
        while reqs or unnamed:
            if unnamed:
                req_to_install = unnamed.pop(0)
            else:
                req_to_install = reqs.pop(0)
            install = True
            if not self.ignore_installed and not req_to_install.editable:
                if req_to_install.check_if_exists():
                    if not self.upgrade:
                        # If we are upgrading, we still need to check the version
                        install = False
            if req_to_install.satisfied_by is not None:
                logger.notify('Requirement already satisfied: %s' % req_to_install)
            elif req_to_install.editable:
                logger.notify('Checking out %s' % req_to_install)
            else:
                if req_to_install.url and req_to_install.url.lower().startswith('file:'):
                    logger.notify('Unpacking %s' % display_path(url_to_filename(req_to_install.url)))
                else:
                    logger.notify('Downloading/unpacking %s' % req_to_install)
            logger.indent += 2
            is_bundle = False
            try:
                if req_to_install.editable:
                    location = req_to_install.build_location(self.src_dir)
                    req_to_install.source_dir = location
                    req_to_install.update_editable()
                    req_to_install.run_egg_info()
                elif install:
                    location = req_to_install.build_location(self.build_dir)
                    ## FIXME: is the existance of the checkout good enough to use it?  I'm don't think so.
                    unpack = True
                    if not os.path.exists(os.path.join(location, 'setup.py')):
                        ## FIXME: this won't upgrade when there's an existing package unpacked in `location`
                        if req_to_install.url is None:
                            url = finder.find_requirement(req_to_install, upgrade=self.upgrade)
                        else:
                            ## FIXME: should req_to_install.url already be a link?
                            url = Link(req_to_install.url)
                            assert url
                        if url:
                            try:
                                self.unpack_url(url, location)
                            except urllib2.HTTPError, e:
                                logger.fatal('Could not install requirement %s because of error %s'
                                             % (req_to_install, e))
                                raise InstallationError(
                                    'Could not install requirement %s because of HTTP error %s for URL %s'
                                    % (req_to_install, e, url))
                        else:
                            unpack = False
                    if unpack:
                        is_bundle = req_to_install.is_bundle
                        if is_bundle:
                            for subreq in req_to_install.bundle_requirements():
                                reqs.append(subreq)
                                self.add_requirement(subreq)
                            req_to_install.move_bundle_files(self.build_dir, self.src_dir)
                        else:
                            req_to_install.source_dir = location
                            req_to_install.run_egg_info()
                            req_to_install.assert_source_matches_version()
                            f = open(req_to_install.delete_marker_filename, 'w')
                            f.write(DELETE_MARKER_MESSAGE)
                            f.close()
                if not is_bundle:
                    ## FIXME: shouldn't be globally added:
                    finder.add_dependency_links(req_to_install.dependency_links)
                    ## FIXME: add extras in here:
                    for req in req_to_install.requirements():
                        try:
                            name = pkg_resources.Requirement.parse(req).project_name
                        except ValueError, e:
                            ## FIXME: proper warning
                            logger.error('Invalid requirement: %r (%s) in requirement %s' % (req, e, req_to_install))
                            continue
                        if self.has_requirement(name):
                            ## FIXME: check for conflict
                            continue
                        subreq = InstallRequirement(req, req_to_install)
                        reqs.append(subreq)
                        self.add_requirement(subreq)
                    if req_to_install.name not in self.requirements:
                        self.requirements[req_to_install.name] = req_to_install
                else:
                    req_to_install.remove_temporary_source()
            finally:
                logger.indent -= 2

    def unpack_url(self, link, location):
        if link.scheme == 'svn' or link.scheme == 'svn+ssh':
            self.svn_checkout(link, location)
            return
        dir = tempfile.mkdtemp()
        if link.url.lower().startswith('file:'):
            source = url_to_filename(link.url)
            content_type = mimetypes.guess_type(source)
            self.unpack_file(source, location, content_type, link)
            return
        md5_hash = link.md5_hash
        target_url = link.url.split('#', 1)[0]
        target_file = None
        if os.environ.get('PYINSTALL_DOWNLOAD_CACHE'):
            target_file = os.path.join(os.environ['PYINSTALL_DOWNLOAD_CACHE'],
                                       urllib.quote(target_url, ''))
        if (target_file and os.path.exists(target_file)
            and os.path.exists(target_file+'.content-type')):
            fp = open(target_file+'.content-type')
            content_type = fp.read().strip()
            fp.close()
            if md5_hash:
                download_hash = md5()
                fp = open(target_file, 'rb')
                while 1:
                    chunk = fp.read(4096)
                    if not chunk:
                        break
                    download_hash.update(chunk)
                fp.close()
            temp_location = target_file
            logger.notify('Using download cache from %s' % target_file)
        else:
            try:
                resp = urllib2.urlopen(target_url)
            except urllib2.HTTPError, e:
                logger.fatal("HTTP error %s while getting %s" % (e.code, link))
                raise
            except IOError, e:
                # Typically an FTP error
                logger.fatal("Error %s while getting %s" % (e, link))
                raise
            content_type = resp.info()['content-type']
            filename = link.filename
            ext = splitext(filename)
            if not ext:
                ext = mimetypes.guess_extension(content_type)
                filename += ext
            temp_location = os.path.join(dir, filename)
            fp = open(temp_location, 'wb')
            if md5_hash:
                download_hash = md5()
            try:
                total_length = int(resp.info()['content-length'])
            except (ValueError, KeyError):
                total_length = 0
            downloaded = 0
            show_progress = total_length > 40*1000 or not total_length
            show_url = link.show_url
            try:
                if show_progress:
                    ## FIXME: the URL can get really long in this message:
                    if total_length:
                        logger.start_progress('Downloading %s (%s): ' % (show_url, format_size(total_length)))
                    else:
                        logger.start_progress('Downloading %s (unknown size): ' % show_url)
                else:
                    logger.notify('Downloading %s' % show_url)
                logger.debug('Downloading from URL %s' % link)
                while 1:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if show_progress:
                        if not total_length:
                            logger.show_progress('%s' % format_size(downloaded))
                        else:
                            logger.show_progress('%3i%%  %s' % (100*downloaded/total_length, format_size(downloaded)))
                    if md5_hash:
                        download_hash.update(chunk)
                    fp.write(chunk)
                fp.close()
            finally:
                if show_progress:
                    logger.end_progress('%s downloaded' % format_size(downloaded))
        if md5_hash:
            download_hash = download_hash.hexdigest()
            if download_hash != md5_hash:
                logger.fatal("MD5 hash of the package %s (%s) doesn't match the expected hash %s!"
                             % (link, download_hash, md5_hash))
                raise InstallationError('Bad MD5 hash for package %s' % link)
        self.unpack_file(temp_location, location, content_type, link)
        if target_file and target_file != temp_location:
            logger.notify('Storing download in cache at %s' % display_path(target_file))
            shutil.copyfile(temp_location, target_file)
            fp = open(target_file+'.content-type', 'w')
            fp.write(content_type)
            fp.close()
        os.unlink(temp_location)

    def unpack_file(self, filename, location, content_type, link):
        if (content_type == 'application/zip'
            or filename.endswith('.zip')
            or filename.endswith('.pybundle')):
            self.unzip_file(filename, location, flatten=not filename.endswith('.pybundle'))
        elif (content_type == 'application/x-gzip'
              or tarfile.is_tarfile(filename)
              or splitext(filename)[1].lower() in ('.tar', '.tar.gz', '.tar.bz2', '.tgz')):
            self.untar_file(filename, location)
        elif (content_type.startswith('text/html')
              and is_svn_page(file_contents(filename))):
            # We don't really care about this
            self.svn_checkout(link.url, location)
        else:
            ## FIXME: handle?
            ## FIXME: magic signatures?
            logger.fatal('Cannot unpack file %s (downloaded from %s, content-type: %s); cannot detect archive format'
                         % (filename, location, content_type))
            raise InstallationError('Cannot determine archive format of %s' % location)

    def unzip_file(self, filename, location, flatten=True):
        """Unzip the file (zip file located at filename) to the destination
        location"""
        if not os.path.exists(location):
            os.makedirs(location)
        zipfp = open(filename, 'rb')
        try:
            zip = zipfile.ZipFile(zipfp)
            leading = has_leading_dir(zip.namelist()) and flatten
            for name in zip.namelist():
                data = zip.read(name)
                fn = name
                if leading:
                    fn = split_leading_dir(name)[1]
                fn = os.path.join(location, fn)
                dir = os.path.dirname(fn)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                if fn.endswith('/'):
                    # A directory
                    if not os.path.exists(fn):
                        os.makedirs(fn)
                else:
                    fp = open(fn, 'wb')
                    try:
                        fp.write(data)
                    finally:
                        fp.close()
        finally:
            zipfp.close()

    def untar_file(self, filename, location):
        """Untar the file (tar file located at filename) to the destination location"""
        if not os.path.exists(location):
            os.makedirs(location)
        if filename.lower().endswith('.gz') or filename.lower().endswith('.tgz'):
            mode = 'r:gz'
        elif filename.lower().endswith('.bz2'):
            mode = 'r:bz2'
        elif filename.lower().endswith('.tar'):
            mode = 'r'
        else:
            logger.warn('Cannot determine compression type for file %s' % filename)
            mode = 'r:*'
        tar = tarfile.open(filename, mode)
        try:
            leading = has_leading_dir([member.name for member in tar.getmembers()])
            for member in tar.getmembers():
                fn = member.name
                if leading:
                    fn = split_leading_dir(fn)[1]
                path = os.path.join(location, fn)
                if member.isdir():
                    if not os.path.exists(path):
                        os.makedirs(path)
                else:
                    fp = tar.extractfile(member)
                    if not os.path.exists(os.path.dirname(path)):
                        os.makedirs(os.path.dirname(path))
                    destfp = open(path, 'wb')
                    try:
                        shutil.copyfileobj(fp, destfp)
                    finally:
                        destfp.close()
                    fp.close()
        finally:
            tar.close()

    def svn_checkout(self, url, location):
        """Check out the svn repository at the url to the destination location"""
        if '#' in url:
            url = url.split('#', 1)[0]
        logger.notify('Checking out svn repository %s to %s' % (url, location))
        logger.indent += 2
        try:
            ## FIXME: not sure that --force is good, but it is needed
            ## when installing directly (not via a requirement),
            ## because the destination directory already exists.
            call_subprocess(['svn', 'checkout', '--force', url, location],
                            filter_stdout=self._filter_svn, show_stdout=False)
        finally:
            logger.indent -= 2

    def _filter_svn(self, line):
        return (Logger.INFO, line)

    def install(self, install_options):
        """Install everything in this set (after having downloaded and unpacked the packages)"""
        requirements = sorted(self.requirements.values(), key=lambda p: p.name.lower())
        logger.notify('Installing collected packages: %s' % (', '.join([req.name for req in requirements])))
        logger.indent += 2
        try:
            for requirement in self.requirements.values():
                if requirement.satisfied_by is not None:
                    # Already installed
                    continue
                requirement.install(install_options)
                requirement.remove_temporary_source()
        finally:
            logger.indent -= 2

    def create_bundle(self, bundle_filename):
        ## FIXME: can't decide which is better; zip is easier to read
        ## random files from, but tar.bz2 is smaller and not as lame a
        ## format.

        ## FIXME: this file should really include a manifest of the
        ## packages, maybe some other metadata files.  It would make
        ## it easier to detect as well.
        zip = zipfile.ZipFile(bundle_filename, 'w', zipfile.ZIP_DEFLATED)
        svn_dirs = []
        for dir, basename in (self.build_dir, 'build'), (self.src_dir, 'src'):
            dir = os.path.normcase(os.path.abspath(dir))
            for dirpath, dirnames, filenames in os.walk(dir):
                svn_url = svn_rev = None
                if '.svn' in dirnames:
                    for svn_dir in svn_dirs:
                        if dirpath.startswith(svn_dir):
                            # svn-checkout.txt already in parent directory
                            break
                    else:
                        svn_url, svn_rev = _get_svn_info(os.path.join(dir, dirpath))
                        svn_dirs.append(dirpath)
                    dirnames.remove('.svn')
                for dirname in dirnames:
                    dirname = os.path.join(dirpath, dirname)
                    name = self._clean_zip_name(dirname, dir)
                    zip.writestr(basename + '/' + name + '/', '')
                for filename in filenames:
                    filename = os.path.join(dirpath, filename)
                    name = self._clean_zip_name(filename, dir)
                    zip.write(filename, basename + '/' + name)
                if svn_url:
                    name = os.path.join(dirpath, 'svn-checkout.txt')
                    name = self._clean_zip_name(name, dir)
                    zip.writestr(basename + '/' + name, self._svn_checkout_text(svn_url, svn_rev))
        zip.writestr('pyinstall-manifest.txt', self.bundle_requirements())
        zip.close()
        # Unlike installation, this will always delete the build directories
        logger.info('Removing temporary build dir %s and source dir %s'
                    % (self.build_dir, self.src_dir))
        for dir in self.build_dir, self.src_dir:
            if os.path.exists(dir):
                shutil.rmtree(dir)

    def _svn_checkout_text(self, svn_url, svn_rev):
        return ('# This was an svn checkout; to make it a checkout again run:\n'
                'svn checkout --force -r %s %s .\n' % (svn_rev, svn_url))

    BUNDLE_HEADER = '''\
# This is a pyinstall bundle file, that contains many source packages
# that can be installed as a group.  You can install this like:
#     pyinstall this_file.zip
# The rest of the file contains a list of all the packages included:
'''

    def bundle_requirements(self):
        parts = [self.BUNDLE_HEADER]
        for req in sorted(
            [req for req in self.requirements.values()
             if not req.comes_from],
            key=lambda x: x.name):
            parts.append('%s==%s\n' % (req.name, req.installed_version))
        parts.append('# These packages were installed to satisfy the above requirements:\n')
        for req in sorted(
            [req for req in self.requirements.values()
             if req.comes_from],
            key=lambda x: x.name):
            parts.append('%s==%s\n' % (req.name, req.installed_version))
        ## FIXME: should we do something with self.unnamed_requirements?
        return ''.join(parts)

    def _clean_zip_name(self, name, prefix):
        assert name.startswith(prefix+'/'), (
            "name %r doesn't start with prefix %r" % (name, prefix))
        name = name[len(prefix)+1:]
        name = name.replace(os.path.sep, '/')
        return name

class HTMLPage(object):
    """Represents one page, along with its URL"""

    ## FIXME: these regexes are horrible hacks:
    _homepage_re = re.compile(r'<th>\s*home\s*page', re.I)
    _download_re = re.compile(r'<th>\s*download\s+url', re.I)
    ## These aren't so aweful:
    _rel_re = re.compile("""<[^>]*\srel\s*=\s*['"]?([^'">]+)[^>]*>""", re.I)
    _href_re = re.compile('href=(?:"([^"]*)"|\'([^\']*)\'|([^>\\s\\n]*))', re.I|re.S)

    def __init__(self, content, url, headers=None):
        self.content = content
        self.url = url
        self.headers = headers

    def __str__(self):
        return self.url

    @classmethod
    def get_page(cls, link, req, cache=None, skip_archives=True):
        url = link.url
        url = url.split('#', 1)[0]
        if cache.too_many_failures(url):
            return None
        if url.lower().startswith('svn'):
            logger.debug('Cannot look at svn URL %s' % link)
            return None
        if cache is not None:
            inst = cache.get_page(url)
            if inst is not None:
                return inst
        try:
            if skip_archives:
                if cache is not None:
                    if cache.is_archive(url):
                        return None
                filename = link.filename
                for bad_ext in ['.tar', '.tar.gz', '.tar.bz2', '.tgz', '.zip']:
                    if filename.endswith(bad_ext):
                        content_type = cls._get_content_type(url)
                        if content_type.lower().startswith('text/html'):
                            break
                        else:
                            logger.debug('Skipping page %s because of Content-Type: %s' % (link, content_type))
                            if cache is not None:
                                cache.set_is_archive(url)
                            return None
            logger.debug('Getting page %s' % url)
            resp = urllib2.urlopen(url)
            real_url = resp.geturl()
            headers = resp.info()
            inst = cls(resp.read(), real_url, headers)
        except (urllib2.HTTPError, urllib2.URLError, socket.timeout, socket.error), e:
            desc = str(e)
            if isinstance(e, socket.timeout):
                log_meth = logger.warn
                level =1
                desc = 'timed out'
            elif isinstance(e, urllib2.URLError):
                log_meth = logger.warn
                if hasattr(e, 'reason') and isinstance(e.reason, socket.timeout):
                    desc = 'timed out'
                    level = 1
                else:
                    level = 2
            elif isinstance(e, urllib2.HTTPError) and e.code == 404:
                ## FIXME: notify?
                log_meth = logger.info
                level = 2
            else:
                log_meth = logger.warn
                level = 1
            log_meth('Could not fetch URL %s: %s' % (link, desc))
            log_meth('Will skip URL %s when looking for download links for %s' % (link.url, req))
            if cache is not None:
                cache.add_page_failure(url, level)
            return None
        if cache is not None:
            cache.add_page([url, real_url], inst)
        return inst

    @staticmethod
    def _get_content_type(url):
        """Get the Content-Type of the given url, using a HEAD request"""
        scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
        if scheme == 'http':
            ConnClass = httplib.HTTPConnection
        elif scheme == 'https':
            ConnClass = httplib.HTTPSConnection
        else:
            ## FIXME: some warning or something?
            ## assertion error?
            return ''
        if query:
            path += '?' + query
        conn = ConnClass(netloc)
        try:
            conn.request('HEAD', path, headers={'Host': netloc})
            resp = conn.getresponse()
            if resp.status != 200:
                ## FIXME: doesn't handle redirects
                return ''
            return resp.getheader('Content-Type') or ''
        finally:
            conn.close()

    @property
    def links(self):
        """Yields all links in the page"""
        for match in self._href_re.finditer(self.content):
            url = match.group(1) or match.group(2) or match.group(3)
            yield Link(urlparse.urljoin(self.url, url), self)

    def rel_links(self):
        for url in self.explicit_rel_links():
            yield url
        for url in self.scraped_rel_links():
            yield url

    def explicit_rel_links(self, rels=('homepage', 'download')):
        """Yields all links with the given relations"""
        for match in self._rel_re.finditer(self.content):
            found_rels = match.group(1).lower().split()
            for rel in rels:
                if rel in found_rels:
                    break
            else:
                continue
            match = self._href_re.search(match.group(0))
            if not match:
                continue
            url = match.group(1) or match.group(2) or match.group(3)
            yield Link(urlparse.urljoin(self.url, url), self)

    def scraped_rel_links(self):
        for regex in (self._homepage_re, self._download_re):
            match = regex.search(self.content)
            if not match:
                continue
            href_match = self._href_re.search(self.content, pos=match.end())
            if not href_match:
                continue
            url = match.group(1) or match.group(2) or match.group(3)
            if not url:
                continue
            url = urlparse.urljoin(self.url, url)
            yield Link(url, self)

class PageCache(object):
    """Cache of HTML pages"""

    failure_limit = 3

    def __init__(self):
        self._failures = {}
        self._pages = {}
        self._archives = {}

    def too_many_failures(self, url):
        return self._failures.get(url, 0) >= self.failure_limit

    def get_page(self, url):
        return self._pages.get(url)

    def is_archive(self, url):
        return self._archives.get(url, False)

    def set_is_archive(self, url, value=True):
        self._archives[url] = value

    def add_page_failure(self, url, level):
        self._failures[url] = self._failures.get(url, 0)+level

    def add_page(self, urls, page):
        for url in urls:
            self._pages[url] = page

class Link(object):

    def __init__(self, url, comes_from=None):
        self.url = url
        self.comes_from = comes_from
    
    def __str__(self):
        if self.comes_from:
            return '%s (from %s)' % (self.url, self.comes_from)
        else:
            return self.url

    def __repr__(self):
        return '<Link %s>' % self

    @property
    def filename(self):
        url = self.url
        url = url.split('#', 1)[0]
        url = url.split('?', 1)[0]
        url = url.rstrip('/')
        name = posixpath.basename(url)
        assert name, (
            'URL %r produced no filename' % url)
        return name

    @property
    def scheme(self):
        return urlparse.urlsplit(self.url)[0]

    @property
    def path(self):
        return urlparse.urlsplit(self.url)[2]

    def splitext(self):
        return splitext(posixpath.basename(self.path.rstrip('/')))

    _egg_fragment_re = re.compile(r'#egg=([^&]*)')

    @property
    def egg_fragment(self):
        match = self._egg_fragment_re.search(self.url)
        if not match:
            return None
        return match.group(1)

    _md5_re = re.compile(r'md5=([a-f0-9]+)')

    @property
    def md5_hash(self):
        match = self._md5_re.search(self.url)
        if match:
            return match.group(1)
        return None

    @property
    def show_url(self):
        return posixpath.basename(self.url.split('#', 1)[0].split('?', 1)[0])

############################################################
## Writing freeze files


def write_freeze(filename, requirement, find_links, find_tags=False):
    if filename == '-':
        logger.move_stdout_to_stderr()
    dependency_links = []
    if filename == '-':
        f = sys.stdout
    else:
        ## FIXME: should be possible to overwrite requirement file
        logger.notify('Writing frozen requirements to %s' % filename)
        f = open(filename, 'w')
    for dist in pkg_resources.working_set:
        if dist.has_metadata('dependency_links.txt'):
            dependency_links.extend(dist.get_metadata_lines('dependency_links.txt'))
    for link in find_links:
        if '#egg=' in link:
            dependency_links.append(link)
    for link in find_links:
        f.write('-f %s\n' % link)
    installations = {}
    for dist in pkg_resources.working_set:
        if dist.key in ('setuptools', 'pyinstall', 'python'):
            ## FIXME: also skip virtualenv?
            continue
        req = FrozenRequirement.from_dist(dist, dependency_links, find_tags=find_tags)
        installations[req.name] = req
    if requirement:
        req_f = open(requirement)
        for line in req_f:
            if not line or line.strip().startswith('#'):
                f.write(line)
                continue
            elif line.startswith('-e') or line.startswith('--editable'):
                if line.startswith('-e'):
                    line = line[2:].strip()
                else:
                    line = line[len('--editable'):].strip().lstrip('=')
                line_req = InstallRequirement.from_editable(line)
            elif (line.startswith('-r') or line.startswith('--requirement')
                  or line.startswith('-Z') or line.startswith('--always-unzip')):
                logger.debug('Skipping line %r' % line.strip())
                continue
            else:
                line_req = InstallRequirement.from_line(line)
            if not line_req.name:
                logger.notify("Skipping line because it's not clear what it would install: %s"
                              % line.strip())
                continue
            if line_req.name not in installations:
                logger.warn("Requirement file contains %s, but that package is not installed"
                            % line.strip())
                continue
            f.write(str(installations[line_req.name]))
            del installations[line_req.name]
        f.write('## The following requirements were added by pyinstall --freeze:\n')
    for installation in sorted(installations.values(), key=lambda x: x.name):
        f.write(str(installation))
    if filename != '-':
        logger.notify('Put requirements in %s' % filename)
        f.close()

class FrozenRequirement(object):

    def __init__(self, name, req, editable, comments=()):
        self.name = name
        self.req = req
        self.editable = editable
        self.comments = comments

    _rev_re = re.compile(r'-r(\d+)$')
    _date_re = re.compile(r'-(20\d\d\d\d\d\d)$')

    @classmethod
    def from_dist(cls, dist, dependency_links, find_tags=False):
        location = os.path.normcase(os.path.abspath(dist.location))
        comments = []
        if os.path.exists(os.path.join(location, '.svn')):
            editable = True
            req = get_src_requirement(dist, location, find_tags)
            if req is None:
                logger.warn('Could not determine svn location of %s' % location)
                comments.append('## !! Could not determine svn location')
                req = dist.as_requirement()
                editable = False
        else:
            editable = False
            req = dist.as_requirement()
            specs = req.specs
            assert len(specs) == 1 and specs[0][0] == '=='
            version = specs[0][1]
            ver_match = cls._rev_re.search(version)
            date_match = cls._date_re.search(version)
            if ver_match or date_match:
                svn_location = get_svn_location(dist, dependency_links)
                if not svn_location:
                    logger.warn(
                        'Warning: cannot find svn location for %s' % req)
                    comments.append('## FIXME: could not find svn URL in dependency_links for this package:')
                else:
                    comments.append('# Installing as editable to satisfy requirement %s:' % req)
                    if ver_match:
                        rev = ver_match.group(1)
                    else:
                        rev = '{%s}' % date_match.group(1)
                    editable = True
                    req = 'svn+%s@%s#egg=%s' % (svn_location, rev, cls.egg_name(dist))
        return cls(dist.project_name, req, editable, comments)

    @staticmethod
    def egg_name(dist):
        name = dist.egg_name()
        match = re.search(r'-py\d\.\d$', name)
        if match:
            name = name[:match.start()]
        return name

    def __str__(self):
        req = self.req
        if self.editable:
            req = '-e %s' % req
        return '\n'.join(list(self.comments)+[str(req)])+'\n'

def get_svn_location(dist, dependency_links):
    egg_fragment_re = re.compile(r'#egg=(.*)$')
    for url in dependency_links:
        egg_fragment = Link(url).egg_fragment
        if not egg_fragment:
            continue
        if '-' in egg_fragment:
            ## FIXME: will this work when a package has - in the name?
            key = '-'.join(egg_fragment.split('-')[:-1]).lower()
        else:
            key = egg_fragment
        if key == dist.key:
            return url.split('#', 1)[0]
    return None

def get_src_requirement(dist, location, find_tags):
    if not os.path.exists(os.path.join(location, '.svn')):
        logger.warn('cannot determine version of editable source in %s (is not svn checkout)' % location)
        return dist.as_requirement()
    repo = get_svn_url(location)
    if repo is None:
        return None
    parts = repo.split('/')
    ## FIXME: why not project name?
    egg_project_name = dist.egg_name().split('-', 1)[0]
    if parts[-2] in ('tags', 'tag'):
        # It's a tag, perfect!
        return 'svn+%s#egg=%s-%s' % (repo, egg_project_name, parts[-1])
    elif parts[-2] in ('branches', 'branch'):
        # It's a branch :(
        rev = get_svn_revision(location)
        return 'svn+%s@%s#egg=%s%s-r%s' % (repo, rev, dist.egg_name(), parts[-1], rev)
    elif parts[-1] == 'trunk':
        # Trunk :-/
        rev = get_svn_revision(location)
        if find_tags:
            tag_url = '/'.join(parts[:-1]) + '/tags'
            tag_revs = get_tag_revs(tag_url)
            match = find_tag_match(rev, tag_revs)
            if match:
                logger.notify('trunk checkout %s seems to be equivalent to tag %s' % match)
                return 'svn+%s/%s#egg=%s-%s' % (tag_url, match, egg_project_name, match)
        return 'svn+%s@%s#egg=%s-dev' % (repo, rev, dist.egg_name())
    else:
        # Don't know what it is
        logger.warn('svn URL does not fit normal structure (tags/branches/trunk): %s' % repo)
        rev = get_svn_revision(location)
        return '%s@%s#egg=%s-dev' % (repo, rev, egg_project_name)

_svn_url_re = re.compile('url="([^"]+)"')
_svn_rev_re = re.compile('committed-rev="(\d+)"')

def get_svn_revision(location):
    """
    Return the maximum revision for all files under a given location
    """
    # Note: taken from setuptools.command.egg_info
    revision = 0

    for base, dirs, files in os.walk(location):
        if '.svn' not in dirs:
            dirs[:] = []
            continue    # no sense walking uncontrolled subdirs
        dirs.remove('.svn')
        entries_fn = os.path.join(base, '.svn', 'entries')
        if not os.path.exists(entries_fn):
            ## FIXME: should we warn?
            continue
        f = open(entries_fn)
        data = f.read()
        f.close()

        if data.startswith('8') or data.startswith('9'):
            data = map(str.splitlines,data.split('\n\x0c\n'))
            del data[0][0]  # get rid of the '8'
            dirurl = data[0][3]
            revs = [int(d[9]) for d in data if len(d)>9 and d[9]]+[0]
            if revs:
                localrev = max(revs)
            else:
                localrev = 0
        elif data.startswith('<?xml'):
            dirurl = _svn_url_re.search(data).group(1)    # get repository URL
            revs = [int(m.group(1)) for m in _svn_rev_re.finditer(data)]+[0]
            if revs:
                localrev = max(revs)
            else:
                localrev = 0
        else:
            logger.warn("Unrecognized .svn/entries format; skipping %s", base)
            dirs[:] = []
            continue
        if base == location:
            base_url = dirurl+'/'   # save the root url
        elif not dirurl.startswith(base_url):
            dirs[:] = []
            continue    # not part of the same svn tree, skip it
        revision = max(revision, localrev)

    return revision

def get_svn_url(location):
    # In cases where the source is in a subdirectory, not alongside setup.py
    # we have to look up in the location until we find a real setup.py
    orig_location = location
    while not os.path.exists(os.path.join(location, 'setup.py')):
        last_location = location
        location = os.path.dirname(location)
        if location == last_location:
            # We've traversed up to the root of the filesystem without finding setup.py
            logger.warn("Could not find setup.py for directory %s (tried all parent directories)"
                        % orig_location)
            return None
    f = open(os.path.join(location, '.svn', 'entries'))
    data = f.read()
    f.close()
    if data.startswith('8') or data.startswith('9'):
        data = map(str.splitlines,data.split('\n\x0c\n'))
        del data[0][0]  # get rid of the '8'
        return data[0][3]
    elif data.startswith('<?xml'):
        return _svn_url_re.search(data).group(1)    # get repository URL
    else:
        logger.warn("Unrecognized .svn/entries format in %s" % location)
        # Or raise exception?
        return None

def get_tag_revs(svn_tag_url):
    stdout = call_subprocess(
        ['svn', 'ls', '-v', svn_tag_url], show_stdout=False)
    results = []
    for line in stdout.splitlines():
        parts = line.split()
        rev = int(parts[0])
        tag = parts[-1].strip('/')
        results.append((tag, rev))
    return results

def find_tag_match(rev, tag_revs):
    best_match_rev = None
    best_tag = None
    for tag, tag_rev in tag_revs:
        if (tag_rev > rev and
            (best_match_rev is None or best_match_rev > tag_rev)):
            # FIXME: Is best_match > tag_rev really possible?
            # or is it a sign something is wacky?
            best_match_rev = tag_rev
            best_tag = tag
    return best_tag


############################################################
## Requirement files

_scheme_re = re.compile(r'^(http|https|file):', re.I)
_drive_re = re.compile(r'/*([a-z])\|', re.I)
def get_file_content(url, comes_from=None):
    """Gets the content of a file; it may be a filename, file: URL, or
    http: URL.  Returns (location, content)"""
    match = _scheme_re.search(url)
    if match:
        scheme = match.group(1).lower()
        if (scheme == 'file' and comes_from
            and comes_from.startswith('http')):
            raise InstallationError(
                'Requirements file %s references URL %s, which is local'
                % (comes_from, url))
        if scheme == 'file':
            path = url.split(':', 1)[1]
            path = path.replace('\\', '/')
            match = _drive_re.match(path)
            if match:
                path = match.group(1) + ':' + path.split('|', 1)[1]
            path = urllib.unquote(path)
            if path.startswith('/'):
                path = '/' + path.lstrip('/')
            url = path
        else:
            ## FIXME: catch some errors
            resp = urllib2.urlopen(url)
            return resp.geturl(), resp.read()
    f = open(url)
    content = f.read()
    f.close()
    return url, content

def parse_requirements(filename, finder, comes_from=None):
    filename, content = get_file_content(filename, comes_from=comes_from)
    for line_number, line in enumerate(content.splitlines()):
        line_number += 1
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('-r') or line.startswith('--requirement'):
            if line.startswith('-r'):
                req_url = line[2:].strip()
            else:
                req_url = line[len('--requirement'):].strip().strip('=')
            if _scheme_re.search(filename):
                # Relative to a URL
                req_url = urlparse.urljoin(filename, url)
            elif not _scheme_re.search(req_url):
                req_url = os.path.join(os.path.dirname(filename), req_url)
            for item in parse_requirements(req_url, finder, comes_from=filename):
                yield item
        elif line.startswith('-Z') or line.startswith('--always-unzip'):
            # No longer used, but previously these were used in
            # requirement files, so we'll ignore.
            pass
        elif line.startswith('-f') or line.startswith('--find-links'):
            if line.startswith('-f'):
                line = line[2:].strip()
            else:
                line = line[len('--find-links'):].strip().lstrip('=')
            ## FIXME: it would be nice to keep track of the source of
            ## the find_links:
            finder.find_links.append(line)
        else:
            comes_from = '-r %s (line %s)' % (filename, line_number)
            if line.startswith('-e') or line.startswith('--editable'):
                if line.startswith('-e'):
                    line = line[2:].strip()
                else:
                    line = line[len('--editable'):].strip()
                req = InstallRequirement.from_editable(
                    line, comes_from)
            else:
                req = InstallRequirement(line, comes_from)
            yield req

############################################################
## Logging



class Logger(object):

    """
    Logging object for use in command-line script.  Allows ranges of
    levels, to avoid some redundancy of displayed information.
    """

    VERBOSE_DEBUG = logging.DEBUG-1
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    NOTIFY = (logging.INFO+logging.WARN)/2
    WARN = WARNING = logging.WARN
    ERROR = logging.ERROR
    FATAL = logging.FATAL

    LEVELS = [VERBOSE_DEBUG, DEBUG, INFO, NOTIFY, WARN, ERROR, FATAL]

    def __init__(self, consumers):
        self.consumers = consumers
        self.indent = 0
        self.in_progress = None
        self.in_progress_hanging = False

    def debug(self, msg, *args, **kw):
        self.log(self.DEBUG, msg, *args, **kw)
    def info(self, msg, *args, **kw):
        self.log(self.INFO, msg, *args, **kw)
    def notify(self, msg, *args, **kw):
        self.log(self.NOTIFY, msg, *args, **kw)
    def warn(self, msg, *args, **kw):
        self.log(self.WARN, msg, *args, **kw)
    def error(self, msg, *args, **kw):
        self.log(self.WARN, msg, *args, **kw)
    def fatal(self, msg, *args, **kw):
        self.log(self.FATAL, msg, *args, **kw)
    def log(self, level, msg, *args, **kw):
        if args:
            if kw:
                raise TypeError(
                    "You may give positional or keyword arguments, not both")
        args = args or kw
        rendered = None
        for consumer_level, consumer in self.consumers:
            if self.level_matches(level, consumer_level):
                if (self.in_progress_hanging
                    and consumer in (sys.stdout, sys.stderr)):
                    self.in_progress_hanging = False
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                if rendered is None:
                    if args:
                        rendered = msg % args
                    else:
                        rendered = msg
                    rendered = ' '*self.indent + rendered
                if hasattr(consumer, 'write'):
                    consumer.write(rendered+'\n')
                else:
                    consumer(rendered)

    def start_progress(self, msg):
        assert not self.in_progress, (
            "Tried to start_progress(%r) while in_progress %r"
            % (msg, self.in_progress))
        if self.level_matches(self.NOTIFY, self._stdout_level()):
            sys.stdout.write(' '*self.indent + msg)
            sys.stdout.flush()
            self.in_progress_hanging = True
        else:
            self.in_progress_hanging = False
        self.in_progress = msg
        self.last_message = None

    def end_progress(self, msg='done.'):
        assert self.in_progress, (
            "Tried to end_progress without start_progress")
        if self.stdout_level_matches(self.NOTIFY):
            if not self.in_progress_hanging:
                # Some message has been printed out since start_progress
                sys.stdout.write('...' + self.in_progress + msg + '\n')
                sys.stdout.flush()
            else:
                # These erase any messages shown with show_progress (besides .'s)
                logger.show_progress('')
                logger.show_progress('')
                sys.stdout.write(msg + '\n')
                sys.stdout.flush()
        self.in_progress = None
        self.in_progress_hanging = False

    def show_progress(self, message=None):
        """If we are in a progress scope, and no log messages have been
        shown, write out another '.'"""
        if self.in_progress_hanging:
            if message is None:
                sys.stdout.write('.')
                sys.stdout.flush()
            else:
                if self.last_message:
                    padding = ' ' * max(0, len(self.last_message)-len(message))
                else:
                    padding = ''
                sys.stdout.write('\r%s%s%s%s' % (' '*self.indent, self.in_progress, message, padding))
                sys.stdout.flush()
                self.last_message = message

    def stdout_level_matches(self, level):
        """Returns true if a message at this level will go to stdout"""
        return self.level_matches(level, self._stdout_level())

    def _stdout_level(self):
        """Returns the level that stdout runs at"""
        for level, consumer in self.consumers:
            if consumer is sys.stdout:
                return level
        return self.FATAL

    def level_matches(self, level, consumer_level):
        """
        >>> l = Logger()
        >>> l.level_matches(3, 4)
        False
        >>> l.level_matches(3, 2)
        True
        >>> l.level_matches(slice(None, 3), 3)
        False
        >>> l.level_matches(slice(None, 3), 2)
        True
        >>> l.level_matches(slice(1, 3), 1)
        True
        >>> l.level_matches(slice(2, 3), 1)
        False
        """
        if isinstance(level, slice):
            start, stop = level.start, level.stop
            if start is not None and start > consumer_level:
                return False
            if stop is not None or stop <= consumer_level:
                return False
            return True
        else:
            return level >= consumer_level

    @classmethod
    def level_for_integer(cls, level):
        levels = cls.LEVELS
        if level < 0:
            return levels[0]
        if level >= len(levels):
            return levels[-1]
        return levels[level]

    def move_stdout_to_stderr(self):
        to_remove = []
        to_add = []
        for consumer_level, consumer in self.consumers:
            if consumer == sys.stdout:
                to_remove.append((consumer_level, consumer))
                to_add.append((consumer_level, sys.stderr))
        for item in to_remove:
            self.consumers.remove(item)
        self.consumers.extend(to_add)


def call_subprocess(cmd, show_stdout=True,
                    filter_stdout=None, cwd=None,
                    raise_on_returncode=True,
                    command_level=Logger.DEBUG, command_desc=None,
                    extra_environ=None):
    if command_desc is None:
        cmd_parts = []
        for part in cmd:
            if ' ' in part or '\n' in part or '"' in part or "'" in part:
                part = '"%s"' % part.replace('"', '\\"')
            cmd_parts.append(part)
        command_desc = ' '.join(cmd_parts)
    if show_stdout:
        stdout = None
    else:
        stdout = subprocess.PIPE
    logger.log(command_level, "Running command %s" % command_desc)
    env = os.environ.copy()
    if extra_environ:
        env.update(extra_environ)
    try:
        proc = subprocess.Popen(
            cmd, stderr=subprocess.STDOUT, stdin=None, stdout=stdout,
            cwd=cwd, env=env)
    except Exception, e:
        logger.fatal(
            "Error %s while executing command %s" % (e, command_desc))
        raise
    all_output = []
    if stdout is not None:
        stdout = proc.stdout
        while 1:
            line = stdout.readline()
            if not line:
                break
            line = line.rstrip()
            all_output.append(line + '\n')
            if filter_stdout:
                level = filter_stdout(line)
                if isinstance(level, tuple):
                    level, line = level
                logger.log(level, line)
                if not logger.stdout_level_matches(level):
                    logger.show_progress()
            else:
                logger.info(line)
    else:
        returned_stdout, returned_stderr = proc.communicate()
        all_output = [returned_stdout or '']
    proc.wait()
    if proc.returncode:
        if raise_on_returncode:
            if all_output:
                logger.notify('Complete output from command %s:' % command_desc)
                logger.notify('\n'.join(all_output) + '\n----------------------------------------')
            raise InstallationError(
                "Command %s failed with error code %s"
                % (command_desc, proc.returncode))
        else:
            logger.warn(
                "Command %s had error code %s"
                % (command_desc, proc.returncode))
    if stdout is not None:
        return ''.join(all_output)


_svn_url_re = re.compile(r'URL: (.+)')
_svn_revision_re = re.compile(r'Revision: (.+)')

def _get_svn_info(dir):
    """Returns (url, revision), where both are strings"""
    assert not dir.rstrip('/').endswith('.svn'), 'Bad directory: %s' % dir
    output = call_subprocess(['svn', 'info', dir], show_stdout=False,
                             extra_environ={'LANG': 'C'})
    match = _svn_url_re.search(output)
    if not match:
        logger.warn('Cannot determine URL of svn checkout %s' % display_path(dir))
        logger.info('Output that cannot be parsed: \n%s' % output)
        return 'unknown', 'unknown'
    url = match.group(1).strip()
    match = _svn_revision_re.search(output)
    if not match:
        logger.warn('Cannot determine revision of svn checkout %s' % display_path(dir))
        logger.info('Output that cannot be parsed: \n%s' % output)
        return url, 'unknown'
    return url, match.group(1)

############################################################
## Utility functions

def is_svn_page(html):
    """Returns true if the page appears to be the index page of an svn repository"""
    return (re.search(r'<title>[^<]*Revision \d+:', html)
            and re.search(r'Powered by (?:<a[^>]*?>)?Subversion', html, re.I))

def file_contents(filename):
    fp = open(filename, 'rb')
    try:
        return fp.read()
    finally:
        fp.close()

def split_leading_dir(path):
    path = str(path)
    path = path.lstrip('/').lstrip('\\')
    if '/' in path and (('\\' in path and path.find('/') < path.find('\\'))
                        or '\\' not in path):
        return path.split('/', 1)
    elif '\\' in path:
        return path.split('\\', 1)
    else:
        return path, ''

def has_leading_dir(paths):
    """Returns true if all the paths have the same leading path name
    (i.e., everything is in one subdirectory in an archive)"""
    common_prefix = None
    for path in paths:
        prefix, rest = split_leading_dir(path)
        if not prefix:
            return False
        elif common_prefix is None:
            common_prefix = prefix
        elif prefix != common_prefix:
            return False
    return True

def format_size(bytes):
    if bytes > 1000*1000:
        return '%.1fMb' % (bytes/1000.0/1000)
    elif bytes > 10*1000:
        return '%iKb' % (bytes/1000)
    elif bytes > 1000:
        return '%.1fKb' % (bytes/1000.0)
    else:
        return '%ibytes' % bytes

_normalize_re = re.compile(r'[^a-z]', re.I)

def normalize_name(name):
    return _normalize_re.sub('-', name.lower())

def display_path(path):
    """Gives the display value for a given path, making it relative to cwd
    if possible."""
    path = os.path.normcase(os.path.abspath(path))
    if path.startswith(os.getcwd() + os.path.sep):
        path = '.' + path[len(os.getcwd()):]
    return path

def parse_editable(editable_req):
    """Parses svn+http://blahblah@rev#egg=Foobar into a requirement
    (Foobar) and a URL"""
    match = re.search(r'(?:#|#.*?&)egg=([^&]*)', editable_req)
    if not match or not match.group(1):
        raise InstallationError(
            '--editable=%s is not the right format; it must have #egg=Package'
            % editable_req)
    req = match.group(1)
    ## FIXME: use package_to_requirement?
    match = re.search(r'^(.*?)(?:-dev|-\d.*)', req)
    if match:
        # Strip off -dev, -0.2, etc.
        req = match.group(1)
    url = editable_req
    if url.lower().startswith('svn:'):
        url = 'svn+' + url
    if '+' not in url:
        raise InstallationError(
            '--editable=%s should be formatted with svn+URL' % editable_req)
    vc_type = url.split('+', 1)[0].lower()
    if vc_type != 'svn':
        raise InstallationError(
            'For --editable=%s only svn (svn+URL) is currently supported' % editable_req)
    return req, url

def backup_dir(dir, ext='.bak'):
    """Figure out the name of a directory to back up the given dir to
    (adding .bak, .bak2, etc)"""
    n = 1
    extension = ext
    while os.path.exists(dir + extension):
        n += 1
        extension = ext + str(n)
    return dir + extension

def ask(message, options):
    """Ask the message interactively, with the given possible responses"""
    while 1:
        response = raw_input(message)
        response = response.strip().lower()
        if response not in options:
            print 'Your response (%r) was not one of the expected responses: %s' % (
                response, ', '.join(options))
        else:
            return response

def open_logfile_append(filename):
    """Open the named log file in append mode.

    If the file already exists, a separator will also be printed to
    the file to separate past activity from current activity.
    """
    exists = os.path.exists(filename)
    log_fp = open(filename, 'a')
    if exists:
        print >> log_fp, '-'*60
        print >> log_fp, '%s run on %s' % (sys.argv[0], time.strftime('%c'))
    return log_fp

def is_url(name):
    """Returns true if the name looks like a URL"""
    if ':' not in name:
        return False
    scheme = name.split(':', 1)[0].lower()
    return scheme in ('http', 'https', 'file', 'ftp')

def is_filename(name):
    if (splitext(name)[1].lower() in ('.zip', '.tar.gz', '.tar.bz2', '.tgz', '.tar', '.pybundle')
        and os.path.exists(name)):
        return True
    if os.path.sep not in name and '/' not in name:
        # Doesn't have any path components, probably a requirement like 'Foo'
        return False
    return True

_drive_re = re.compile('^([a-z]):', re.I)
_url_drive_re = re.compile('^([a-z])[|]', re.I)

def filename_to_url(filename):
    """
    Convert a path to a file: URL.  The path will be made absolute.
    """
    filename = os.path.normcase(os.path.abspath(filename))
    url = urllib.quote(filename)
    if _drive_re.match(url):
        url = url[0] + '|' + url[2:]
    url = url.replace(os.path.sep, '/')
    url = url.lstrip('/')
    return 'file:///' + url

def url_to_filename(url):
    """
    Convert a file: URL to a path.
    """
    assert url.startswith('file:'), (
        "You can only turn file: urls into filenames (not %r)" % url)
    filename = url[len('file:'):].lstrip('/')
    filename = urllib.unquote(filename)
    if _url_drive_re.match(filename):
        filename = filename[0] + ':' + filename[2:]
    else:
        filename = '/' + filename
    return filename

def get_requirement_from_url(url):
    """Get a requirement from the URL, if possible.  This looks for #egg
    in the URL"""
    link = Link(url)
    egg_info = link.egg_fragment
    if not egg_info:
        egg_info = splitext(link.filename)[0]
    return package_to_requirement(egg_info)

def package_to_requirement(package_name):
    """Translate a name like Foo-1.2 to Foo==1.3"""
    match = re.search(r'^(.*?)(-dev|-\d.*)', package_name)
    if match:
        name = match.group(1)
        version = match.group(2)
    else:
        name = package_name
        version = ''
    if version:
        return '%s==%s' % (name, version)
    else:
        return name


def splitext(path):
    """Like os.path.splitext, but take off .tar too"""
    base, ext = posixpath.splitext(path)
    if base.lower().endswith('.tar'):
        ext = base[-4:] + ext
        base = base[:-4]
    return base, ext

class _Inf(object):
    """I am bigger than everything!"""
    def __cmp__(self, a):
        if self is a:
            return 0
        return 1
    def __repr__(self):
        return 'Inf'
Inf = _Inf()
del _Inf

if __name__ == '__main__':
    main()
